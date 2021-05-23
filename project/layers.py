import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np

tfd = tfp.distributions

normal = tfp.layers.DistributionLambda(
    lambda t: tfd.Normal(
        loc=t[..., :1],
        scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:])
    )
)

# https://en.wikipedia.org/wiki/Poisson_distribution#Related_distributions
poisson_approximation = tfp.layers.DistributionLambda(
    lambda t: tfd.Normal(
        loc=1e-3 + tf.math.softplus(0.05 * t),
        scale=tf.math.sqrt(1e-3 + tf.math.softplus(0.05 * t))
    )
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
#https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
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


class LocationScaleMixture(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        # inputs.shape => (batch, time, params=7)
        assert inputs.shape[2] == 13
        normal_loc = inputs[..., 0:1]
        normal_scale = 1e-3 + tf.math.softplus(0.05 * inputs[..., 1:2])
        # Limit the tails of the Student t
        student_df = 5.0 + tf.math.softplus(0.05 * inputs[..., 2:3])
        student_loc = inputs[..., 3:4]
        student_scale = 1e-3 + tf.math.softplus(inputs[..., 4:5])
        laplace_loc = inputs[..., 5:6]
        laplace_scale = 1e-3 + tf.math.softplus(0.05 * inputs[..., 6:7])
        uniform_low = inputs[..., 7:8]
        uniform_high = inputs[..., 8:9]
        logits = tf.math.softplus(inputs[..., 9:])

        # sum_i p_i = 1
        # Pr(X ~ D_i(...)) = p_i
        cat = tfd.Categorical(logits=logits)[..., None]

        normal = tfd.Normal(
            loc=normal_loc,
            scale=normal_scale,
            validate_args=True
        )

        studentT = tfd.StudentT(
            df=student_df,
            loc=student_loc,
            scale=student_scale,
            validate_args=True
        )

        laplace = tfd.Laplace(
            loc=laplace_loc,
            scale=laplace_scale,
            validate_args=True
        )

        uniform = tfd.Uniform(
            low=uniform_low,
            high=uniform_high
        )

        mixture = tfd.Mixture(
            cat=cat,
            components=[
                normal,
                studentT,
                laplace,
                uniform
            ]
        )
        return mixture


class HiddenMarkovModel(Layer):
    def __init__(self,
                 number_states, forecast_length,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_states = number_states
        self.forecast_length = forecast_length
        self.rate_prior = tfd.Normal(loc=tf.zeros([self.num_states]),
                                     scale=tf.ones([self.num_states]))
        self.hmm = None

    def call(self, inputs, *args, **kwargs):
        # inputs.shape => (batch, params)
        # (params=self.num_states + self.num_states**2 + 2 * self.num_states)
        initial_state_logits = inputs[..., : self.num_states]
        transition_logits = inputs[..., self.num_states:
                                        self.num_states + self.num_states ** 2]

        # Convert transition_logits into a square matrix
        transition_logits = tf.reshape(transition_logits,
                                       (-1,
                                        self.num_states,
                                        self.num_states))

        # Get the start of the distribution's parameters
        distribution_index = self.num_states + self.num_states ** 2
        normal_loc = inputs[..., distribution_index:distribution_index + self.num_states]
        normal_scale = inputs[..., distribution_index + self.num_states:]
        normal_scale = 1e-3 + tf.math.softplus(0.05 * normal_scale)

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

        self.hmm = tfd.HiddenMarkovModel(
            initial_distribution=initial_distribution,
            transition_distribution=transition_distribution,
            observation_distribution=observation_distribution,
            num_steps=self.forecast_length
        )

        return self.hmm


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    https://keras.io/examples/generative/vae/"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon