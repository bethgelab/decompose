from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import ParameterInfo
from decompose.distributions.distribution import Distribution
from decompose.distributions.cenLaplace import CenLaplace
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.cenDoubleLomaxAlgorithms import CenDoubleLomaxAlgorithms
from decompose.distributions.distribution import Properties


class CenDoubleLomax(Distribution):
    def __init__(self,
                 algorithms: Type[Algorithms] = CenDoubleLomaxAlgorithms,
                 alpha: Tensor = None,
                 beta: Tensor = None,
                 tau: Tensor = None,
                 properties: Properties = None) -> None:
        parameters = {"alpha": alpha, "beta": beta, "tau": tau}
        Distribution.__init__(self,
                              algorithms=algorithms,
                              parameters=parameters,
                              properties=properties)

    def parameterInfo(self,
                      shape: Tuple[int, ...] = (1,),
                      latentShape: Tuple[int, ...] = ()) -> ParameterInfo:
        initializers = {
            "alpha": (shape, True),
            "beta": (shape, True),
            "tau": (latentShape + shape, True)
        }  # type: Dict[str, Tensor]
        return(initializers)

    @parameterProperty
    def mu(self) -> Tensor:
        mu = tf.zeros_like(self.alpha)
        return(mu)

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

    def cond(self) -> CenLaplace:
        tau = self.tau
        name = self.name + "Cond"
        properties = Properties(name=name,
                                drawType=self.drawType,
                                updateType=self.updateType,
                                persistent=False)
        cond = CenLaplace(beta=1./tau,
                          properties=properties)
        return(cond)

    @property
    def shape(self) -> Tuple[int, ...]:
        return(tuple(self.mu.get_shape().as_list()))

    @property
    def latentShape(self) -> Tuple[int, ...]:
        ndims = len(self.tau.get_shape().as_list()) - len(self.shape)
        return(tuple(self.tau.get_shape().as_list()[:ndims]))
