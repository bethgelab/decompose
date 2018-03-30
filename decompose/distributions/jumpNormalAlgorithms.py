from typing import Tuple, Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms
from decompose.distributions.chopin2011 import rtnorm


class JumpNormalAlgorithms(Algorithms):

    @classmethod
    def alpha(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mu = parameters["mu"]
        tau = parameters["tau"]
        nu = parameters["nu"]
        beta = parameters["beta"]

        sigma = 1./tf.sqrt(tau)
        lam = 1./beta

        muStd = tf.constant(0., dtype=mu.dtype)
        sigmaStd = tf.constant(1., dtype=mu.dtype)
        stdNorm = tf.contrib.distributions.Normal(loc=muStd, scale=sigmaStd)

        c0 = lam*(mu-nu) + stdNorm.log_cdf((nu-(mu+sigma**2*lam))/sigma)
        c1 = -lam*(mu-nu) + stdNorm.log_cdf(-(nu-(mu-sigma**2*lam))/sigma)
        c = tf.reduce_logsumexp([c0, c1], axis=0)
        f = (mu-nu)*lam

        norm = tf.distributions.Normal(loc=mu+sigma**2*lam, scale=sigma)

        alpha = tf.exp(f + norm.log_cdf(nu) - c)
        return(alpha)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mu = parameters["mu"]
        tau = parameters["tau"]
        nu = parameters["nu"]
        beta = parameters["beta"]

        lam = 1./beta
        mode = tf.zeros_like(mu) * tf.zeros_like(mu)
        mode = tf.where(tf.logical_and(tf.greater(nu, mu),
                                       tf.less(mu+lam/tau, nu)),
                        mu+lam/tau,
                        mode)
        mode = tf.where(tf.logical_and(tf.greater(nu, mu),
                                       tf.greater_equal(mu+lam/tau, nu)),
                        nu,
                        mode)
        mode = tf.where(tf.logical_and(tf.less_equal(nu, mu),
                                       tf.greater(mu-lam/tau, nu)),
                        mu-lam/tau,
                        mode)
        mode = tf.where(tf.logical_and(tf.less_equal(nu, mu),
                                       tf.less_equal(mu-lam/tau, nu)),
                        nu,
                        mode)
        return(mode)

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        mu = parameters["mu"]
        tau = parameters["tau"]
        nu = parameters["nu"]
        beta = parameters["beta"]

        shape = tf.concat(((nSamples,), tf.shape(mu)), 0)
        ones = tf.ones(shape=shape, dtype=mu.dtype)
        mu = tf.reshape(mu*ones, (-1,))
        tau = tf.reshape(tau*ones, (-1,))
        nu = tf.reshape(nu*ones, (-1,))
        beta = tf.reshape(beta*ones, (-1,))
        inf = tf.ones_like(mu)*tf.constant(np.inf, dtype=mu.dtype)

        # sample from which side of the distribution to sample from
        alpha = cls.alpha(parameters)
        rUni = tf.random_uniform(shape=tf.shape(mu), dtype=mu.dtype)
        isRight = tf.greater(rUni, alpha)

        # sample from the left side
        rl = rtnorm(a=-inf, b=nu, mu=mu+1./(tau*beta), sigma=1./tf.sqrt(tau))

        # sample from the right side
        rr = rtnorm(a=nu, b=inf, mu=mu-1./(tau*beta), sigma=1./tf.sqrt(tau))

        # pick the samples randomly
        r = tf.where(isRight, rr, rl)

        # reshaping the samples
        r = tf.reshape(r, shape)
        return(r)

    @classmethod
    def logConstant(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mu = parameters["mu"]
        tau = parameters["tau"]
        nu = parameters["nu"]
        beta = parameters["beta"]

        sigma = 1./tf.sqrt(tau)
        zero = tf.constant(0., dtype=mu.dtype)
        one = tf.constant(1., dtype=mu.dtype)
        stdNormal = tf.distributions.Normal(loc=zero, scale=one)
        l0 = (mu-nu)/beta + stdNormal.log_cdf((nu-(mu+1./(beta*tau)))/sigma)
        l1 = -(mu-nu)/beta + stdNormal.log_cdf(-(nu-(mu-1./(beta*tau)))/sigma)
        log_c = tf.reduce_logsumexp((l0, l1))
        return(log_c)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        logProb = cls.llh(parameters, data)
        prob = tf.exp(logProb)
        return(prob)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> Tensor:
        mu = parameters["mu"]
        tau = parameters["tau"]
        nu = parameters["nu"]
        beta = parameters["beta"]

        logConstant = cls.logConstant(parameters)
        sign = tf.sign(data-nu)
        loc = mu-1./(tau*beta)*sign
        scale = 1./tf.sqrt(tau)
        normal = tf.distributions.Normal(loc=loc, scale=scale)
        logProb = normal.log_prob(value=data) - (mu-nu)*sign/beta - logConstant
        return(logProb)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        raise NotImplemented

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        return({})
