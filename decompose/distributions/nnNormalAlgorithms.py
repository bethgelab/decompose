from typing import Tuple, Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms
from decompose.distributions.chopin2011 import rtnorm


class NnNormalAlgorithms(Algorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        # TODO: sample nSample many samples
        mu, tau = parameters["mu"], parameters["tau"]
        sigma = tf.sqrt(1./tau)
        a = tf.zeros_like(mu)
        b = tf.ones_like(a)*np.inf

        r = rtnorm(a=a, b=b, mu=mu, sigma=sigma)
        r = tf.expand_dims(r, axis=-1)
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
        pdf = (1.
               / sigma
               * norm.pdf(value=((data-mu)/sigma))
               / norm.cdf(value=(mu/sigma)))
        pdf = tf.where(tf.greater(data, 0.), pdf, tf.zeros_like(pdf))
        return(pdf)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        raise NotImplemented("Not yet implemented")

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
        return(llh)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        return({})
