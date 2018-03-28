from typing import Dict
import tensorflow as tf
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
        exp = tf.distributions.Exponential(rate=1./beta)
        pdf = exp.prob(value=data)
        return(pdf)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        beta = tf.reduce_mean(tf.abs(data), axis=0)
        updatedParameters = {"beta": beta}
        return(updatedParameters)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> float:
        beta = parameters["beta"]
        exp = tf.distributions.Exponential(rate=1./beta)
        llh = exp.log_prob(value=data)
        return(llh)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        return({})
