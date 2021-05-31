import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

normal = tfp.layers.DistributionLambda(
    lambda t: tfd.Normal(
        loc=t[..., :1],
        scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]),
        name='normal'
    ),
    name='normal'
)

laplace = tfp.layers.DistributionLambda(
    lambda t: tfd.Laplace(
        loc=t[..., :1],
        scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]),
        name='laplace'
    ),
    name='laplace'
)

# https://en.wikipedia.org/wiki/Poisson_distribution#Related_distributions
poisson_approximation = tfp.layers.DistributionLambda(
    lambda t: tfd.Normal(
        loc=t,
        scale=tf.math.sqrt(1e-3 + tf.math.softplus(0.05 * t)),
        name='poisson_approx'
    ),
    name='poisson_approx'
)


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=0.0001),
            reinterpreted_batch_ndims=1)),
    ])


variational_normal = tf.keras.Sequential([
    tfp.layers.DenseVariational(1 + 1,
                                posterior_mean_field,
                                prior_trainable),
    normal
])


class T2V(Layer):
    # https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        self.P = self.add_weight(name='P',
                                 shape=(input_shape[1], self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        self.w = self.add_weight(name='w',
                                 shape=(input_shape[1], 1),
                                 initializer='uniform',
                                 trainable=True)
        self.p = self.add_weight(name='p',
                                 shape=(input_shape[1], 1),
                                 initializer='uniform',
                                 trainable=True)
        super(T2V, self).build(input_shape)

    def call(self, x):
        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)

        return K.concatenate([sin_trans, original], -1)


class StudentT(tfp.layers.DistributionLambda):
    """Layer that returns a Student t with min_df"""
    def __init__(self, min_df=1.0, **kwargs):
        super(StudentT, self).__init__(
            lambda t: StudentT.new(min_df, t),
            ** kwargs
        )
        self.min_df = min_df

    @staticmethod
    def new(min_df, inputs):
        """Builds a mixture model based off four loc-scale distributions"""
        # inputs.shape => (batch, time, params=3)
        assert inputs.shape[2] == 3
        # Limit the tails of the Student t
        return tfd.StudentT(
            df=min_df + tf.math.softplus(0.05 * inputs[..., 0:1]),
            loc=inputs[..., 1:2],
            scale=1e-3 + tf.math.softplus(inputs[..., 2:3]),
            name='student_t',
            validate_args=True
        )

class LocationScaleMixture(tfp.layers.DistributionLambda):
    """Layer that returns a Location-Scale mixture model"""
    def __init__(self, min_df=1.0, **kwargs):
        super(LocationScaleMixture, self).__init__(
            lambda t: LocationScaleMixture.new(min_df, t),
            **kwargs
        )
        self.min_df = min_df

    @staticmethod
    def new(min_df, inputs):
        """Builds a mixture model based off four loc-scale distributions"""
        # inputs.shape => (batch, time, params=13)
        assert inputs.shape[2] == 11
        normal_loc = inputs[..., 0:1]
        normal_scale = 1e-3 + tf.math.softplus(0.05 * inputs[..., 1:2])
        # Limit the tails of the Student-t
        student_df = min_df + tf.math.softplus(0.05 * inputs[..., 2:3])
        student_loc = inputs[..., 3:4]
        student_scale = 1e-3 + tf.math.softplus(0.05 * inputs[..., 4:5])
        laplace_loc = inputs[..., 5:6]
        laplace_scale = 1e-3 + tf.math.softplus(0.05 * inputs[..., 6:7])
        logits = inputs[...,8:]
        #concentration = 1e-3 + tf.math.softplus(inputs[..., 9:])

        # prior = tfd.Dirichlet(
        #     concentration=concentration
        # )
        #
        # logits = tf.math.log(prior.sample())

        # sum_i p_i = 1
        # Pr(X ~ D_i(...)) = p_i
        cat = tfd.Categorical(
            logits=logits,
            name='mixture_categories'
        )[..., None]

        normal = tfd.Normal(
            loc=normal_loc,
            scale=normal_scale,
            validate_args=True,
            name='normal_component'
        )

        studentT = tfd.StudentT(
            df=student_df,
            loc=student_loc,
            scale=student_scale,
            validate_args=True,
            name='student_t_component'
        )

        laplace = tfd.Laplace(
            loc=laplace_loc,
            scale=laplace_scale,
            validate_args=True,
            name='laplace_component'
        )

        return tfd.Mixture(
            cat=cat,
            components=[
                normal,
                studentT,
                laplace
            ],
            name='location_scale_mixture_model'
        )


class HiddenMarkovModel(tfp.layers.DistributionLambda):
    """Layer that takes parameters as inputs and returns an HMM distribution. """
    def __init__(self, number_states, forecast_length, **kwargs):
        super(HiddenMarkovModel, self).__init__(
            # DistributionLambda handles the `call` method -- we just specify
            # a static function for building an HMM distribution from params.
            lambda t: HiddenMarkovModel.new(number_states, forecast_length, t),
            activity_regularizer=self.apply_prior,
            **kwargs)
        self.num_states = number_states
        self.forecast_length = forecast_length

        # Trainable prior.
        self.prior_scale = tfp.util.TransformedVariable(1., bijector=tfb.Exp(),
                                                        name='hmm_prior_scale')
        self.observation_scale_prior = tfd.HalfNormal(self.prior_scale,
                                                      name='hmm_half_normal_prior')

    @staticmethod
    def new(num_states, forecast_length, inputs):
        # inputs.shape => (batch, params)
        # (params=self.num_states + self.num_states**2 + 2 * self.num_states)
        initial_state_logits = inputs[..., : num_states]
        transition_logits = inputs[..., num_states:
                                        num_states + num_states ** 2]

        # Convert transition_logits into a square matrix
        transition_logits = tf.reshape(transition_logits,
                                       (-1,
                                        num_states,
                                        num_states))

        # Get the start of the distribution's parameters
        distribution_index = num_states + num_states ** 2
        normal_loc = inputs[..., distribution_index:distribution_index + num_states]
        normal_scale = inputs[..., distribution_index + num_states:]
        normal_scale = 1e-3 + tf.math.softplus(normal_scale)

        # Create the input distributions to HMM
        initial_distribution = tfd.Categorical(
            logits=initial_state_logits,
            name='initial_logits',
            validate_args=True
        )

        transition_distribution = tfd.Categorical(
            logits=transition_logits,
            name='transition_logits',
            validate_args=True
        )

        observation_distribution = tfd.Normal(
            loc=normal_loc,
            scale=normal_scale,
            name='observation_distribution',
            validate_args=True
        )

        return tfd.HiddenMarkovModel(
            initial_distribution=initial_distribution,
            transition_distribution=transition_distribution,
            observation_distribution=observation_distribution,
            num_steps=forecast_length,
            name='hidden_markov_model'
        )

    def apply_prior(self, hmm):
        # Necessary hack to make Keras activity regularization work with TFP Distribution outputs.
        hmm = getattr(hmm, '_tfp_distribution', hmm)

        # If the layer was applied to an input of shape [batch_shape, N], then
        #  - `hmm.batch_shape` is `[batch_shape]`
        #  - `hmm.observation_distribution.scale` has shape `[batch_shape, 2]`
        #  - `self.observation_scale_prior.log_prob(...)` has shape `[batch_shape]`
        #  - we will return the prior log_prob scaled by batch_shape
        #    (which is the right thing to do, because Keras will divide the
        #     returned value by batch_shape)
        return tf.reduce_sum(
            -self.observation_scale_prior.log_prob(hmm.observation_distribution.scale)
        )


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    https://keras.io/examples/generative/vae/"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
