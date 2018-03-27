from typing import Tuple, Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms


class NormalAlgorithms(Algorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        mu, tau = parameters["mu"], parameters["tau"]
        shape = tf.concat((mu.shape, (nSamples,)), 0)
        mu = tf.expand_dims(mu, -1)
        tau = tf.expand_dims(tau, -1)

        dtype = mu.dtype
        muStd = tf.constant(0., dtype=dtype)
        sigmaStd = tf.constant(1., dtype=dtype)
        stdNorm = tf.distributions.Normal(loc=muStd, scale=sigmaStd)
        r = stdNorm.sample(sample_shape=shape)
        r = r*tf.sqrt(1./tau) + mu
        return(r)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mu = parameters["mu"]
        return(mu)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        mu, tau = parameters["mu"], parameters["tau"]
        norm = tf.distributions.Normal(loc=mu, scale=tf.sqrt(1./tau))
        pdf = norm.pdf(x=data)
        return(pdf)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        mu = tf.reduce_mean(data, axis=-1)
        var = tf.reduce_mean((data-tf.expand_dims(mu, -1))**2, axis=-1)
        tau = 1./var
        updatedParameters = {"mu": mu, "tau": tau}
        return(updatedParameters)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> float:
        mu, tau = parameters["mu"], parameters["tau"]
        norm = tf.distributions.Normal(loc=mu, scale=tf.sqrt(1./tau))
        llh = norm.log_prob(value=data)
        return(llh)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        return({})
