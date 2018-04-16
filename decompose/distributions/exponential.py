from typing import Tuple, Any, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.exponentialAlgorithms import ExponentialAlgorithms


class Exponential(Distribution):
    def __init__(self,
                 beta: Tensor = tf.constant([1.]),
                 name: str = "NA",
                 algorithms: Type[Algorithms] = ExponentialAlgorithms,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 persistent: bool = True) -> None:
        Distribution.__init__(self,
                              shape=beta.shape,
                              latentShape=(),
                              name=name,
                              algorithms=algorithms,
                              drawType=drawType,
                              dtype=beta.dtype,
                              updateType=updateType,
                              persistent=persistent)
        self._init({"beta": beta})

    @staticmethod
    def initializers(shape: Tuple[int, ...] = (1,),
                     latentShape: Tuple[int, ...] = (),
                     dtype: np.dtype = np.float32) -> Dict[str, Tensor]:
        dtype = tf.as_dtype(dtype)
        one = tf.constant(1., dtype=dtype)
        exponential = tf.distributions.Exponential(rate=one)
        initializers = {
            "beta": exponential.sample(sample_shape=shape)
        }  # type: Dict[str, Tensor]
        return(initializers)

    @parameterProperty
    def beta(self) -> Tensor:
        return(self.__beta)

    @beta.setter("beta")
    def beta(self, beta: Tensor):
        self.__beta = beta

    @property
    @classmethod
    def nonNegative(cls) -> bool:
        return(True)

    @property
    def homogenous(self) -> bool:
        return(False)

    def cond(self) -> Distribution:
        return(self)
