from likelihoods import Gaussian

import tensorflow as tf
from gpflow import settings


def reparameterize(mean, var, z, full_cov=False):
    if var is None:
        return mean

    if full_cov is False:
        return mean + z * (var + settings.jitter) ** 0.5

    else:
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2] # var is SNND
        mean = tf.transpose(mean, (0, 2, 1))  # SND -> SDN
        var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
        I = settings.jitter * tf.eye(N, dtype=settings.float_type)[None,None, :, :] # 11NN
        chol = tf.cholesky(var + I)  # SDNN
        z_res = tf.transpose(z, [0, 2, 1])[:, :, :, None]  # SND->SDN1
        f = mean + tf.matmul(chol, z_res)[:, :,:, 0]  # SDN(1)
        return tf.transpose(f, (0, 2, 1)) # SND

