from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import ParameterInfo
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.cenNnFullyElasticNetAlgorithms import CenNnFullyElasticNetAlgorithms
from decompose.distributions.cenNnFullyElasticNetCond import CenNnFullyElasticNetCond
from decompose.distributions.distribution import Properties


class CenNnFullyElasticNet(Distribution):
    def __init__(self,
                 algorithms: Type[Algorithms] = CenNnFullyElasticNetAlgorithms,
                 b: Tensor = None,
                 mu: Tensor = None,
                 betaExponential: Tensor = None,
                 tau: Tensor = None,
                 alpha: Tensor = None,
                 beta: Tensor = None,
                 tauLomax: Tensor = None,
                 properties: Properties = None) -> None:
        parameters = {"b": b,
                      "mu": mu, "tau": tau,
                      "betaExponential": betaExponential,
                      "alpha": alpha, "beta": beta, "tauLomax": tauLomax}
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
            "alpha": (shape, True),
            "beta": (shape, True),
            "tauLomax": (latentShape + shape, True)

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
    def muLomax(self) -> Tensor:
        muLomax = tf.zeros_like(self.alpha)
        return(muLomax)

    @parameterProperty
    def alpha(self) -> tf.Tensor:
        return(self.__alpha)

    @alpha.setter(name="alpha")
    def alpha(self, alpha: tf.Tensor) -> None:
        self.__alpha = alpha

    @parameterProperty
    def beta(self) -> tf.Tensor:
        return(self.__beta)

    @beta.setter(name="beta")
    def beta(self, beta: tf.Tensor) -> None:
        self.__beta = beta

    @parameterProperty
    def tau(self) -> Tensor:
        return(self.__tau)

    @tau.setter(name="tau")
    def tau(self, tau: Tensor) -> None:
        self.__tau = tau

    @property
    def nonNegative(self) -> bool:
        return(True)

    @property
    def homogenous(self) -> bool:
        return(False)

    def cond(self) -> CenNnFullyElasticNetCond:
        b = self.b
        mu = self.mu
        tau = self.tau
        betaExponential = self.betaExponential
        tauLomax = self.tauLomax
        b = tf.ones_like(tauLomax)*b
        mu = tf.ones_like(tauLomax)*mu
        tau = tf.ones_like(tauLomax)*tau
        betaExponential = tf.ones_like(tauLomax)*betaExponential
        name = self.name + "Cond"
        properties = Properties(name=name,
                                drawType=self.drawType,
                                updateType=self.updateType,
                                persistent=False)
        cond = CenNnFullyElasticNetCond(b=b, mu=mu, tau=tau,
                                        betaExponential=betaExponential,
                                        beta=1./tauLomax,
                                        properties=properties)
        return(cond)

    @property
    def shape(self) -> Tuple[int, ...]:
        return(tuple(self.mu.get_shape().as_list()))

    @property
    def latentShape(self) -> Tuple[int, ...]:

        return(())
