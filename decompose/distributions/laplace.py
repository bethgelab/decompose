from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.laplaceAlgorithms import LaplaceAlgorithms


class Laplace(Distribution):
    def __init__(self,
                 mu: Tensor = tf.constant([0.]),
                 beta: Tensor = tf.constant([1.]),
                 name: str = "NA",
                 algorithms: Type[Algorithms] = LaplaceAlgorithms,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 persistent: bool = True) -> None:
        Distribution.__init__(self,
                              shape=mu.shape,
                              latentShape=(),
                              name=name,
                              drawType=drawType,
                              dtype=mu.dtype,
                              updateType=updateType,
                              persistent=persistent,
                              algorithms=algorithms)
        self._init({"mu": mu,
                    "beta": beta})

    @staticmethod
    def initializers(shape: Tuple[int, ...] = (1,),
                     latentShape: Tuple[int, ...] = (),
                     dtype: np.dtype = np.float32) -> Dict[str, Tensor]:
        dtype = tf.as_dtype(dtype)
        zero = tf.constant(0., dtype=dtype)
        one = tf.constant(1., dtype=dtype)
        normal = tf.distributions.Normal(loc=zero, scale=one)
        exponential = tf.distributions.Exponential(rate=one)
        initializers = {
            "mu": normal.sample(sample_shape=shape),
            "beta": exponential.sample(sample_shape=shape)
        }  # type: Dict[str, Tensor]
        return(initializers)

    @parameterProperty
    def mu(self) -> Tensor:
        return(self.__mu)

    @mu.setter(name="mu")
    def mu(self, mu: Tensor) -> None:
        self.__mu = mu

    @parameterProperty
    def beta(self) -> Tensor:
        return(self.__beta)

    @beta.setter(name="beta")
    def beta(self, beta: Tensor) -> None:
        self.__beta = beta

    @property
    @classmethod
    def nonNegative(cls) -> bool:
        return(False)

    @property
    def homogenous(self) -> bool:
        return(False)

    def cond(self) -> Distribution:
        return(self)
