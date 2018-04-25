from typing import Tuple, Any, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import ParameterInfo
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.exponentialAlgorithms import ExponentialAlgorithms
from decompose.distributions.distribution import Properties


class Exponential(Distribution):
    def __init__(self,
                 algorithms: Type[Algorithms] = ExponentialAlgorithms,
                 beta: Tensor = None,
                 properties: Properties = None) -> None:
        parameters = {"beta": beta}
        Distribution.__init__(self,
                              algorithms=algorithms,
                              parameters=parameters,
                              properties=properties)

    def parameterInfo(self,
                      shape: Tuple[int, ...] = (1,),
                      latentShape: Tuple[int, ...] = ()) -> ParameterInfo:
        initializers = {
            "beta": (shape, True)
        }  # type: Dict[str, Tensor]
        return(initializers)

    @parameterProperty
    def beta(self) -> Tensor:
        return(self.__beta)

    @beta.setter("beta")
    def beta(self, beta: Tensor):
        self.__beta = beta

    @property
    def nonNegative(self) -> bool:
        return(True)

    @property
    def homogenous(self) -> bool:
        return(False)

    def cond(self) -> Distribution:
        return(self)

    @property
    def shape(self) -> Tuple[int, ...]:
        return(tuple(self.beta.get_shape().as_list()))

    @property
    def latentShape(self) -> Tuple[int, ...]:
        return(())
