from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import ParameterInfo
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.jumpNormalAlgorithms import JumpNormalAlgorithms
from decompose.distributions.distribution import Properties


class JumpNormal(Distribution):
    def __init__(self,
                 algorithms: Type[Algorithms] = JumpNormalAlgorithms,
                 mu: Tensor = None,
                 tau: Tensor = None,
                 nu: Tensor = None,
                 beta: Tensor = None,
                 properties: Properties = None) -> None:
        parameters = {"mu": mu, "tau": tau, "nu": nu, "beta": beta}
        Distribution.__init__(self,
                              algorithms=algorithms,
                              parameters=parameters,
                              properties=properties)

    def parameterInfo(self,
                      shape: Tuple[int, ...] = (1,),
                      latentShape: Tuple[int, ...] = ())-> ParameterInfo:
        initializers = {
            "mu": (shape, False),
            "tau": (shape, True),
            "nu": (shape, False),
            "beta": (shape, True),
        }  # type: Dict[str, Tensor]
        return(initializers)

    @parameterProperty
    def mu(self) -> Tensor:
        return(self.__mu)

    @mu.setter(name="mu")
    def mu(self, mu: Tensor) -> None:
        self.__mu = mu

    @parameterProperty
    def tau(self) -> Tensor:
        return(self.__tau)

    @tau.setter(name="tau")
    def tau(self, tau: Tensor) -> None:
        self.__tau = tau

    @parameterProperty
    def nu(self) -> Tensor:
        return(self.__nu)

    @nu.setter(name="nu")
    def nu(self, nu: Tensor) -> None:
        self.__nu = nu

    @parameterProperty
    def beta(self) -> Tensor:
        return(self.__beta)

    @beta.setter(name="beta")
    def beta(self, beta: Tensor) -> None:
        self.__beta = beta

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
        return(tuple(self.mu.get_shape().as_list()))

    @property
    def latentShape(self) -> Tuple[int, ...]:
        return(())
