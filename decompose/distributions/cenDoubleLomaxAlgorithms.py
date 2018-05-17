from typing import Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms
from decompose.distributions.lomaxAlgorithms import LomaxAlgorithms


class CenDoubleLomaxAlgorithms(Algorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        alpha, beta = parameters["alpha"], parameters["beta"]
        gamma = tf.distributions.Gamma(concentration=alpha, rate=beta)
        tau = gamma.sample(sample_shape=(nSamples,))
        lap = tf.distributions.Laplace(scale=1./tau,
                                       loc=tf.zeros_like(alpha))
        s = lap.sample()
        return(s)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mode = tf.zeros_like(parameters["alpha"])
        return(mode)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        pdf = LomaxAlgorithms.pdf(parameters=parameters, data=tf.abs(data))/2.
        return(pdf)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> Tensor:
        llh = LomaxAlgorithms.llh(parameters=parameters,
                                  data=tf.abs(data)) - np.log(2.)
        return(llh)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        parameters = LomaxAlgorithms.fit(parameters=parameters,
                                         data=tf.abs(data))
        return(parameters)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        parameters = LomaxAlgorithms.fitLatents(parameters=parameters,
                                                data=tf.abs(data))
        return(parameters)
