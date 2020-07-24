import numpy as np
import tensorflow as tf

from gpflow import logdensities
from gpflow import priors
from gpflow import settings
from gpflow import transforms
from gpflow.quadrature import hermgauss
from gpflow.quadrature import ndiagquad, ndiag_mc


class Likelihood:
    def __init__(self, *args, **kwargs):
        self.num_gauss_hermite_points = 20

    def predict_mean_and_var(self, Fmu, Fvar):
        integrand2 = lambda *X: self.conditional_variance(*X) + tf.square(self.conditional_mean(*X))
        E_y, E_y2 = ndiagquad([self.conditional_mean, integrand2],
                              self.num_gauss_hermite_points,
                              Fmu, Fvar)
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y


class Gaussian(Likelihood):
    def __init__(self, variance=1.0, **kwargs):
        super().__init__(**kwargs)
        self.variance = Param(variance,
                 transform=transforms.Log1pe(),
                 name  = "noise_variance")()

    def logp(self, F, Y):
        return logdensities.gaussian(Y, F, self.variance)

    def conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def predict_density(self, Fmu, Fvar, Y):
        return logdensities.gaussian(Y, Fmu, Fvar + self.variance)

    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance


class Bernoulli(Likelihood):
    def __init__(self, invlink=inv_probit):
        self.invlink = invlink

    def logp(self, F, Y):
        return logdensities.bernoulli(Y, self.invlink(F))

    def predict_mean_and_var(self, Fmu, Fvar):
        if self.invlink is inv_probit:
            p = inv_probit(Fmu / tf.sqrt(1 + Fvar))
            return p, p - tf.square(p)
        else:
            return super().predict_mean_and_var(Fmu, Fvar)

    def predict_density(self, Fmu, Fvar, Y):
        p = self.predict_mean_and_var(Fmu, Fvar)[0]
        return logdensities.bernoulli(Y, p)

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - tf.square(p)
