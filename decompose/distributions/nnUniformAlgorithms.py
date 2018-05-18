from typing import Tuple, Dict
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.algorithms import Algorithms


class NnUniformAlgorithms(Algorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        raise ValueError

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        raise ValueError

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        raise ValueError

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        updatedParameters = {}  # Dict[str, Tensor]
        return(updatedParameters)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> float:
        llh = tf.where(tf.less(data, 0.),
                       -tf.ones_like(data)*np.inf,
                       tf.zeros_like(data))
        return(llh)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        return({})
