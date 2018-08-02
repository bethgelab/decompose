from typing import Dict
import tensorflow as tf
import numpy as np
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms


class ExponentialAlgorithms(Algorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        beta = parameters["beta"]
        exp = tf.distributions.Exponential(rate=1./beta)
        r = exp.sample(sample_shape=(nSamples,))
        return(r)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mode = tf.zeros_like(parameters["beta"])
        return(mode)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        beta = parameters["beta"]
        # TODO: once tensorflow issue #20737 is resolved use the
        # commented out code to calculate the pdf
        # exp = tf.distributions.Exponential(rate=1./beta)
        # pdf = exp.prob(value=data)
        pdf = 1./beta*tf.exp(-data/beta)
        pdf = tf.where(tf.less(data, 0.), tf.zeros_like(pdf), pdf)
        return(pdf)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        beta = tf.reduce_mean(tf.abs(data), axis=0)
        updatedParameters = {"beta": beta}
        return(updatedParameters)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> Tensor:
        beta = parameters["beta"]
        # TODO: once tensorflow issue #20737 is resolved use the
        # commented out code to calculate the llh
        # exp = tf.distributions.Exponential(rate=1./beta)
        # llh = exp.log_prob(value=data)
        llh = -tf.log(beta) - data/beta
        llh = tf.where(data < 0., -tf.ones_like(llh)*np.inf, llh)
        return(llh)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        return({})
