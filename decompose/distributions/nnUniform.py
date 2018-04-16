from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.nnUniformAlgorithms import NnUniformAlgorithms


class NnUniform(Distribution):
    def __init__(self,
                 dummy: Tensor = tf.constant([0.]),
                 name: str = "NA",
                 algorithms: Type[Algorithms] = NnUniformAlgorithms,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 persistent: bool = True) -> None:
        Distribution.__init__(self,
                              shape=dummy.shape,
                              latentShape=(),
                              name=name,
                              drawType=drawType,
                              dtype=dummy.dtype,
                              updateType=updateType,
                              persistent=persistent,
                              algorithms=algorithms)
        self._init({"dummy": dummy})

    @staticmethod
    def initializers(shape: Tuple[int, ...] = (1,),
                     latentShape: Tuple[int, ...] = (),
                     dtype: np.dtype = np.float32) -> Dict[str, Tensor]:
        dtype = tf.as_dtype(dtype)
        one = tf.constant(1., dtype=dtype)
        exponential = tf.distributions.Exponential(rate=one)
        initializers = {
            "dummy": exponential.sample(sample_shape=shape)
        }  # type: Dict[str, Tensor]
        return(initializers)

    @parameterProperty
    def dummy(self) -> Tensor:
        return(self.__dummy)

    @dummy.setter(name="dummy")
    def dummy(self, dummy: Tensor):
        self.__dummy = dummy

    @property
    @classmethod
    def nonNegative(cls) -> bool:
        return(True)

    @property
    def homogenous(self) -> bool:
        return(True)

    def cond(self) -> Distribution:
        return(self)
