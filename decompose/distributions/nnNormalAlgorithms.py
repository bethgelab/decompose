from typing import Tuple, Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms
from decompose.distributions.chopin2011 import rtnorm


class NnNormalAlgorithms(Algorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        mu, tau = parameters["mu"], parameters["tau"]
        sigma = tf.sqrt(1./tau)
        a = tf.zeros_like(mu)
        b = tf.ones_like(a)*np.inf
        r = rtnorm(a=a, b=b, mu=mu, sigma=sigma, nSamples=nSamples)
        return(r)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mu = parameters["mu"]
        mode = tf.where(tf.greater(mu, 0.), mu, tf.zeros_like(mu))
        return(mode)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        mu, tau = parameters["mu"], parameters["tau"]
        dtype = mu.dtype

        norm = tf.distributions.Normal(loc=tf.constant(0., dtype=dtype),
                                       scale=tf.constant(1., dtype=dtype))
        sigma = 1./tf.sqrt(tau)
        pdf = (1./sigma
               * norm.prob(value=((data-mu)/sigma))
               / norm.cdf(value=(mu/sigma)))
        pdf = tf.where(tf.less(data, 0.), tf.zeros_like(pdf), pdf)
        return(pdf)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> Tensor:
        mu, tau = parameters["mu"], parameters["tau"]
        dtype = mu.dtype

        norm = tf.distributions.Normal(loc=tf.constant(0., dtype=dtype),
                                       scale=tf.constant(1., dtype=dtype))
        sigma = 1./tf.sqrt(tau)
        llh = (- tf.log(sigma)
               + norm.log_prob(value=((data-mu)/sigma))
               - norm.log_cdf(value=(mu/sigma)))
        data = data*tf.ones_like(llh)
        llh = tf.where(tf.less(data, 0.), -tf.ones_like(llh)*np.inf, llh)
        return(llh)

    @classmethod
    def gradStep(cls, data, mu, tau, v, e):
        swidths = tf.constant(np.array([0., 1e-9, 1e-8, 1e-7, 1e-6, 1e-4, 1e-3,
                                        1e-2, 1e-1, 1e0, 1e1, 1e2,
                                        1e3])[..., None, None],
                              dtype=data.dtype)
        for i in range(5):
            llhs = cls.llh(parameters={"mu": mu,
                                       "tau": 1./(v+e**2-e*mu)},
                           data=data)
            llhs = tf.reduce_mean(llhs, axis=0)
            signGradMu = tf.sign(tf.gradients(llhs, [mu])[0])
            mus = mu+signGradMu*swidths
            taus = 1./(v+e**2-e*mus)
            tauIsNonPositive = tf.less_equal(taus, 0.)
            mus = tf.where(tauIsNonPositive, tf.zeros_like(mus), mus)
            taus = tf.where(tauIsNonPositive, 1./tf.reduce_mean(data, axis=0)*tf.ones_like(taus), taus)
            newLlhs = cls.llh(parameters={"mu": mus,
                                          "tau": taus},
                              data=data[None])
            newLlhs = tf.reduce_mean(newLlhs, axis=-2, keepdims=True)
            argmax = tf.cast(tf.argmax(newLlhs, axis=0)[0], dtype=tf.int32)
            mu = tf.gather(mus[:, 0], argmax)
            mu = tf.diag_part(mu)
            tau = tf.gather(taus[:, 0], argmax)
            tau = tf.diag_part(tau)
        return(mu, tau)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:

        muOld, tauOld = parameters["mu"], parameters["tau"]

        e = tf.reduce_mean(data, axis=0)
        v = tf.reduce_mean((data-e)**2, axis=0)
        s = tf.reduce_mean(((data-e)/tf.sqrt(v))**3, axis=0)

        muOld, tauOld = cls.gradStep(data, muOld, tauOld, v, e)

        mu = (s*v**1.5 + e*v - e**3)/(v-e**2)
        tau = 1./(v+e**2-e*mu)
        mu, tau = cls.gradStep(data, mu, tau, v, e)

        llhOld = cls.llh(parameters={"mu": muOld, "tau": tauOld}, data=data)
        llhOld = tf.reduce_mean(llhOld, axis=0)

        llh = cls.llh(parameters={"mu": mu, "tau": tau}, data=data)
        llh = tf.reduce_mean(llh, axis=0)

        mu = tf.where(tf.greater(llhOld, llh), muOld, mu)
        tau = tf.where(tf.greater(llhOld, llh), tauOld, tau)

        assertTauIs0 = tf.Assert(tf.reduce_all(tf.logical_not(tf.equal(tau, 0.))), [tau], name='nnNormalAlgnorNnNortauIs0')
        assertTauNotPos = tf.Assert(tf.reduce_all(tf.greater(tau, 0.)), [tau], name='nnNormalAlgnorNnNorCenNotPositive')
        with tf.control_dependencies([assertTauIs0, assertTauNotPos]):
            tau = tau + 0.

        updatedParameters = {"mu": mu, "tau": tau}
        return(updatedParameters)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        return({})
