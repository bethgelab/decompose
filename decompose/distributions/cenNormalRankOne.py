from typing import Tuple, Any, Dict, Type
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import ParameterInfo
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.normal import Normal
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.cenNormalRankOneAlgorithms import CenNormalRankOneAlgorithms
from decompose.distributions.distribution import Properties


class CenNormalRankOne(Normal):
    def __init__(self,
                 algorithms: Type[Algorithms] = CenNormalRankOneAlgorithms,
                 tau0: Tensor = None,
                 tau1: Tensor = None,
                 properties: Properties = None) -> None:
        parameters = {"tau0": tau0, "tau1": tau1}
        Distribution.__init__(self,
                              algorithms=algorithms,
                              parameters=parameters,
                              properties=properties)

    def parameterInfo(self,
                      shape: Tuple[int, int] = (5, 5),
                      latentShape: Tuple[int, ...] = ()) -> ParameterInfo:
        initializers = {
            "tau0": (shape[0], True),
            "tau1": (shape[1], True)
        }  # type: Dict[str, Tensor]
        return(initializers)

    @property
    def mu(self) -> Tuple[Tensor, Tensor]:
        mu0 = tf.zeros_like(self.tau0)
        mu1 = tf.zeros_like(self.tau1)
        return((mu0, mu1))

    @parameterProperty
    def tau0(self) -> Tensor:
        return(self.__tau0)

    @tau0.setter(name="tau0")
    def tau0(self, tau0: Tensor):
        self.__tau0 = tau0

    @parameterProperty
    def tau1(self) -> Tensor:
        return(self.__tau1)

    @tau1.setter(name="tau1")
    def tau1(self, tau1: Tensor):
        self.__tau1 = tau1

    @property
    def nonNegative(self) -> bool:
        return(False)

    @property
    def homogenous(self) -> bool:
        return(False)

    def cond(self) -> Distribution:
        return(self)

    @property
    def shape(self) -> Tuple[int, ...]:
        return(tuple(self.tau0.get_shape().as_list())
               + tuple(self.tau1.get_shape().as_list()))

    @property
    def latentShape(self) -> Tuple[int, ...]:
        return(())
