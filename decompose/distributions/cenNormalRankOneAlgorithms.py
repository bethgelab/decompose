from typing import Tuple, Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms


class CenNormalRankOneAlgorithms(Algorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        tau0, tau1 = parameters["tau0"], parameters["tau"]
        tau = tf.matmul(tau0[..., None], tau1[None, ...])
        norm = tf.distributions.Normal(loc=tf.zeros_like(tau),
                                       scale=1./tf.sqrt(tau1))
        r = norm.sample(sample_shape=(nSamples,))
        return(r)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        tau0, tau1 = parameters["tau0"], parameters["tau1"]
        M = tau0.get_shape().as_list()[0]
        N = tau1.get_shape().as_list()[0]
        mode = tf.zeros(shape=(M, N), dtype=tau0.dtype)
        return(mode)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        tau0, tau1 = parameters["tau0"], parameters["tau1"]
        tau = tf.matmul(tau0[..., None], tau1[None, ...])
        norm = tf.distributions.Normal(loc=tf.zeros_like(tau),
                                       scale=tf.sqrt(1./tau))
        pdf = norm.prob(value=data)
        return(pdf)

    @classmethod
    def fitGamma(cls, tau):
        alpha = 0.5/(tf.log(tf.reduce_mean(tau))
                     + 1e-6  # added due to numerical instability
                     - tf.reduce_mean(tf.log(tau)))
        for i in range(20):
            alpha = (1. / (1./alpha
                           + (tf.reduce_mean(tf.log(tau))
                              - tf.log(tf.reduce_mean(tau))
                              + tf.log(alpha)
                              - tf.digamma(alpha))
                           / (alpha**2*(1./alpha
                                        - tf.polygamma(tf.ones_like(alpha),
                                                       alpha)))))

        beta = alpha/tf.reduce_mean(tau)
        return(alpha, beta)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        """Optimal ML update using the EM algorithm."""

        # regularized multiplicative
        M, N = data.get_shape().as_list()
        tau0, tau1 = parameters["tau0"], parameters["tau1"]

        # hyperparameter optimization
        alpha0, beta0 = cls.fitGamma(tau0)
        alpha1, beta1 = cls.fitGamma(tau1)

        # sampling taus
        alphaPost0 = alpha0 + N/2
        betaPost0 = beta0 + tf.matmul(data**2, tau1[..., None])[..., 0]/2
        tau0 = tf.distributions.Gamma(concentration=alphaPost0,
                                      rate=betaPost0).sample(1)[0]
        tau0 = tf.where(tau0 < 1e-6, tf.ones_like(tau0)*1e-6, tau0)

        alphaPost1 = alpha1 + M/2
        betaPost1 = beta1 + tf.matmul(tau0[None, ...], data**2)[0, ...]/2
        tau1 = tf.distributions.Gamma(concentration=alphaPost1,
                                      rate=betaPost1).sample(1)[0]
        tau1 = tf.where(tau1 < 1e-6, tf.ones_like(tau1)*1e-6, tau1)

        # rescaling taus
        normTau0 = tf.norm(tau0)
        normTau1 = tf.norm(tau1)
        normPerFactor = tf.sqrt(normTau0*normTau1)
        tau0 = tau0/normTau0*normPerFactor
        tau1 = tau1/normTau1*normPerFactor

        updatedParameters = {"tau0": tau0, "tau1": tau1}
        return(updatedParameters)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> float:
        tau0, tau1 = parameters["tau0"], parameters["tau1"]
        tau = tf.matmul(tau0[..., None], tau1[None, ...])
        norm = tf.distributions.Normal(loc=tf.zeros_like(tau),
                                       scale=tf.sqrt(1./tau))
        llh = norm.log_prob(value=data)
        return(llh)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        return({})
