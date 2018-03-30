from typing import Tuple, Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms


class NormalAlgorithms(Algorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        mu, tau = parameters["mu"], parameters["tau"]
        norm = tf.distributions.Normal(loc=mu, scale=1./tf.sqrt(tau))
        r = norm.sample(sample_shape=(nSamples,))
        return(r)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mu = parameters["mu"]
        return(mu)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        mu, tau = parameters["mu"], parameters["tau"]
        norm = tf.distributions.Normal(loc=mu, scale=tf.sqrt(1./tau))
        pdf = norm.prob(value=data)
        return(pdf)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        mu = tf.reduce_mean(data, axis=0)
        var = tf.reduce_mean((data-mu)**2, axis=0)
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
