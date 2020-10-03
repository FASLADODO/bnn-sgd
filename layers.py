from .reparameterize import reparameterize

import tensorflow as tf
import numpy as np
from gpflow.conditionals import conditional
from gpflow.kullback_leiblers import gauss_kl
from gpflow import transforms
from gpflow import settings


class Layer:
    def __init__(self):
        pass
    def conditional_ND(self, X, full_cov=False):
        raise NotImplementedError

    def KL(self):
        return tf.cast(0., dtype=settings.float_type)
        

class ESBO_Layer(Layer):
    def __init__(self,kern,Um,Us_sqrt,Z,num_outputs,white=True):
        self.white = white
        self.kern = kern
        self.num_outputs = num_outputs
        self.num_inducing = Z.shape[0]
        self.q_diag = True if Us_sqrt.ndim == 2 else False
        with tf.name_scope("inducing"):
            self.Z  = Param(Z, # MxM
                      name="z")()
            self.Um  = Param(Um, #DxM
                       name="u")()
            if self.q_diag:
                self.Us_sqrt = Param(Us_sqrt, # DxM
                                transforms.positive,
                                name="u_variance")()
            else:
                self.Us_sqrt = Param(Us_sqrt, # DxMxM
                                  transforms.LowerTriangular(Us_sqrt.shape[1],Us_sqrt.shape[0]),
                                  name="u_variance")()

        self.Ku = self.kern.Ksymm(self.Z) + tf.eye(tf.shape(self.Z)[0],dtype=self.Z.dtype)*settings.jitter
        self.Lu = tf.cholesky(self.Ku)
        self.Ku_tiled = tf.tile(self.Ku[None, :, :], [self.num_outputs, 1, 1]) # DxMxM
        self.Lu_tiled = tf.tile(self.Lu[None, :, :], [self.num_outputs, 1, 1])

    def KL(self):
        M = tf.cast(self.num_inducing, settings.float_type)
        B = tf.cast(self.num_outputs, settings.float_type)

        logdet_pcov, logdet_qcov, mahalanobis, trace = self.get_kl_terms(self.Um,tf.transpose(self.Us_sqrt) if self.q_diag else self.Us_sqrt) #scalar, Dx1, Dx1, Dx1
        constant  = -M
        twoKL = logdet_pcov - logdet_qcov + mahalanobis + trace + constant
        kl = 0.5*tf.reduce_sum(twoKL)

        return kl

    def get_kl_terms(self,q_mu,q_sqrt):
        if self.white:
            alpha = q_mu  # MxD
        else:
            alpha = tf.matrix_triangular_solve(self.Lu, q_mu, lower=True)  # MxD

        if self.q_diag:
            Lq = Lq_diag = q_sqrt # MxD
            Lq_full = tf.matrix_diag(tf.transpose(q_sqrt))  # DxMxM
        else:
            Lq = Lq_full = tf.matrix_band_part(q_sqrt, -1, 0)  # force lower triangle # DxMxM
            Lq_diag = tf.transpose(tf.matrix_diag_part(Lq))  # MxD

        mahalanobis = tf.reduce_sum(tf.square(alpha),axis=0)[:,None] # Dx1

        logdet_qcov = tf.reduce_sum(tf.log(tf.square(Lq_diag)),axis=0)[:,None] # Dx1

        if self.white:
            if self.q_diag:
                trace = tf.reduce_sum(tf.square(Lq),axis=0)[:,None] # MxD --> Dx1
            else:
                trace = tf.reduce_sum(tf.square(Lq),axis=[1,2])[:,None] # DxMxM --> Dx1
        else:
            if self.q_diag:
                Lp     = self.Lu
                LpT    = tf.transpose(Lp)  
                Lp_inv = tf.matrix_triangular_solve(Lp, tf.eye(self.num_inducing, dtype=settings.float_type),lower=True)  # MxM
                K_inv  = tf.matrix_diag_part(tf.matrix_triangular_solve(LpT, Lp_inv, lower=False))[:, None]  # MxM -> Mx1
                trace  = tf.reduce_sum(K_inv * tf.square(q_sqrt),axis=0)[:,None] # Mx1*MxD --> Dx1
            else:
                Lp_full = self.Lu_tiled
                LpiLq   = tf.matrix_triangular_solve(Lp_full, Lq_full, lower=True) # DxMxM
                trace   = tf.reduce_sum(tf.square(LpiLq),axis=[1,2])[:,None] # Dx1

        if not self.white:
            log_sqdiag_Lp = tf.log(tf.square(tf.matrix_diag_part(self.Lu)))
            logdet_pcov = tf.reduce_sum(log_sqdiag_Lp)
        else:
            logdet_pcov = 0

        return logdet_pcov, logdet_qcov, mahalanobis, trace
