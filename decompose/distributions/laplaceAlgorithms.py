from typing import Tuple, Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms


class LaplaceAlgorithms(Algorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        mu, beta = parameters["mu"], parameters["beta"]
        norm = tf.distributions.Laplace(loc=mu, scale=beta)
        r = norm.sample(sample_shape=(nSamples,))
        print("r in sample", r)
        return(r)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mu = parameters["mu"]
        return(mu)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        mu, beta = parameters["mu"], parameters["beta"]
        norm = tf.distributions.Laplace(loc=mu, scale=beta)
        pdf = norm.prob(value=data)
        return(pdf)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        M = data.get_shape().as_list()[0]
        mu = tf.contrib.nn.nth_element(tf.transpose(data), M//2)
        beta = tf.reduce_mean(tf.abs(data-mu), axis=0)
        updatedParameters = {"mu": mu, "beta": beta}
        return(updatedParameters)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> Tensor:
        mu, beta = parameters["mu"], parameters["beta"]
        norm = tf.distributions.Laplace(loc=mu, scale=beta)
        llh = norm.log_prob(value=data)
        return(llh)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        return({})
