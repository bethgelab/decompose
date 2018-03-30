from typing import Tuple, Dict
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms
from decompose.distributions.laplaceAlgorithms import LaplaceAlgorithms


class CenLaplaceAlgorithms(Algorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        beta = parameters["beta"]
        mu = tf.zeros_like(beta)
        samples = LaplaceAlgorithms.sample(parameters={"mu": mu, "beta": beta},
                                           nSamples=nSamples)
        return(samples)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        beta = parameters["beta"]
        mode = tf.zeros_like(beta)
        return(mode)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        beta = parameters["beta"]
        mu = tf.zeros_like(beta)
        pdf = LaplaceAlgorithms.pdf(parameters={"mu": mu, "beta": beta},
                                    data=data)
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
        mu = tf.zeros_like(beta)
        llh = LaplaceAlgorithms.llh(parameters={"mu": mu, "beta": beta},
                                    data=data)
        return(llh)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        return({})
