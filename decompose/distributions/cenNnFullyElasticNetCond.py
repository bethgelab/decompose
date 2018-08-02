from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import ParameterInfo
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.cenNnFullyElasticNetAlgorithms import CenNnFullyElasticNetAlgorithms
from decompose.distributions.distribution import Properties


class CenNnFullyElasticNetCond(Distribution):
    def __init__(self,
                 algorithms: Type[Algorithms] = CenNnFullyElasticNetAlgorithms,
                 b: Tensor = None,
                 mu: Tensor = None,
                 tau: Tensor = None,
                 betaExponential: Tensor = None,
                 beta: Tensor = None,
                 properties: Properties = None) -> None:
        parameters = {"b": b,
                      "mu": mu, "tau": tau,
                      "betaExponential": betaExponential,
                      "beta": beta}
        Distribution.__init__(self,
                              algorithms=algorithms,
                              parameters=parameters,
                              properties=properties)

    def parameterInfo(self,
                      shape: Tuple[int, ...] = (1,),
                      latentShape: Tuple[int, ...] = ())-> ParameterInfo:
        initializers = {
            "b": (shape, False),
            "mu": (shape, False),
            "tau": (shape, True),
            "betaExponential": (shape, True),
            "beta": (shape, True),
        }  # type: Dict[str, Tensor]
        return(initializers)

    @parameterProperty
    def b(self) -> Tensor:
        return(self.__b)

    @b.setter(name="b")
    def b(self, b: Tensor):
        self.__b = b

    @parameterProperty
    def mu(self) -> Tensor:
        return(self.__mu)

    @mu.setter(name="mu")
    def mu(self, mu: Tensor):
        self.__mu = mu

    @parameterProperty
    def tau(self) -> Tensor:
        return(self.__tau)

    @tau.setter(name="tau")
    def tau(self, tau: Tensor):
        self.__tau = tau

    @parameterProperty
    def betaExponential(self) -> Tensor:
        return(self.__betaExponential)

    @betaExponential.setter(name="betaExponential")
    def betaExponential(self, betaExponential: Tensor):
        self.__betaExponential = betaExponential

    @parameterProperty
    def beta(self) -> tf.Tensor:
        return(self.__beta)

    @beta.setter(name="beta")
    def beta(self, beta: tf.Tensor) -> None:
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
        return(tuple(self.mu.get_shape().as_list()))

    @property
    def latentShape(self) -> Tuple[int, ...]:
        return(())
