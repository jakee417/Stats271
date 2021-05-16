import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np

tfd = tfp.distributions

poisson = tfp.layers.DistributionLambda(
    lambda t: tfd.Poisson(
        # rate=1e-3 + tf.math.softplus(0.05 * t)
        rate=1e-3 + tf.exp(t)
    )
)

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
