from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import ParameterInfo
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.nnUniformAlgorithms import NnUniformAlgorithms
from decompose.distributions.distribution import Properties


class NnUniform(Distribution):
    def __init__(self,
                 algorithms: Type[Algorithms] = NnUniformAlgorithms,
                 dummy: Tensor = None,
                 properties: Properties = None) -> None:
        parameters = {"dummy": dummy}
        Distribution.__init__(self,
                              algorithms=algorithms,
                              parameters=parameters,
                              properties=properties)

    def parameterInfo(self,
                      shape: Tuple[int, ...] = (1,),
                      latentShape: Tuple[int, ...] = ())-> ParameterInfo:
        initializers = {
            "dummy": (shape, False)
        }  # type: Dict[str, Tensor]
        return(initializers)

    @property
    def shape(self) -> Tuple[int, ...]:
        return(tuple(self.dummy.get_shape().as_list()))

    @property
    def latentShape(self) -> Tuple[int, ...]:
        return(())

    @parameterProperty
    def dummy(self) -> Tensor:
        return(self.__dummy)

    @dummy.setter(name="dummy")
    def dummy(self, dummy: Tensor):
        self.__dummy = dummy

    @property
    def nonNegative(self) -> bool:
        return(True)

    @property
    def homogenous(self) -> bool:
        return(True)

    def cond(self) -> Distribution:
        return(self)
