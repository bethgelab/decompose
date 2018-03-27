from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.uniformAlgorithms import UniformAlgorithms


class Uniform(Distribution):
    def __init__(self,
                 dummy: Tensor = tf.constant([0.]),
                 name: str = "NA",
                 algorithms: Type[Algorithms] = UniformAlgorithms,
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
        initializers = {
            "dummy": tf.constant(np.random.normal(size=shape), dtype=dtype)
        }  # type: Dict[str, Tensor]
        return(initializers)

    @parameterProperty
    def dummy(self) -> Tensor:
        return(self.__dummy)

    @dummy.setter(name="dummy")
    def dummy(self, dummy: Tensor):
        self.__dummy = dummy

    @property
    def nonNegative(self) -> bool:
        return(False)

    @property
    def homogenous(self) -> bool:
        return(True)

    def cond(self) -> Distribution:
        return(self)
