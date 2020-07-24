from .likelihoods import Gaussian
from .reparameterize import BroadcastingLikelihood

import tensorflow as tf
import numpy as np

from gpflow import settings

class BNN:
    def __init__(self, likelihood, diff_layer,pred_layer,
                 num_samples=1,flow_time = 1.0,flow_nsteps = 20,num_data=None):
        self.num_samples = num_samples
        self.num_data = num_data
        self.flow_nsteps = flow_nsteps
        self.flow_time = flow_time
        self.likelihood = BroadcastingLikelihood(likelihood)
        self.diff_layer = diff_layer
        self.pred_layer = pred_layer
        self.sde_solver = EulerMaruyama(f= self.diff_layer.conditional_ND,total_time = self.flow_time,nsteps = self.flow_nsteps)

    def integrate(self, X,num_samples=1,save_intermediate=False):
        N, D = tf.shape(X)[0], tf.shape(X)[1]
        Xt = [] ; Xs = []
        for s in range(num_samples):
            _Xt, _Xs = self.sde_solver.forward(X,save_intermediate)
            Xt.append(_Xt[None,:,:])
            if save_intermediate:
                Xs.append(_Xs[None,:,:,:])

        Xt = tf.concat(Xt,axis=0)
        if save_intermediate:
            Xs = tf.concat(Xs,axis=0)
        return Xt, Xs

    def _build_likelihood(self,X,Y):
        L = tf.reduce_mean(self.E_log_p_Y(X, Y))
        KL = tf.reduce_sum([layer.KL() for layer in [self.diff_layer,self.pred_layer]])
        scale = tf.cast(self.num_data, float_type)
        return L - (KL/scale)

    def predict_f(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=False, S=num_samples)

    def predict_y(self, Xnew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    def predict_density(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)
