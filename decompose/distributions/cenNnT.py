from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import ParameterInfo
from decompose.distributions.nnNormal import NnNormal
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.cenNnTAlgorithms import CenNnTAlgorithms
from decompose.distributions.distribution import Properties


class CenNnT(Distribution):
    def __init__(self,
                 algorithms: Type[Algorithms] = CenNnTAlgorithms,
                 Psi: Tensor = None,
                 nu: Tensor = None,
                 tau: Tensor = None,
                 properties: Properties = None) -> None:
        parameters = {"Psi": Psi, "nu": nu, "tau": tau}
        Distribution.__init__(self,
                              algorithms=algorithms,
                              parameters=parameters,
                              properties=properties)

    def parameterInfo(self,
                      shape: Tuple[int, ...] = (1,),
                      latentShape: Tuple[int, ...] = ()) -> ParameterInfo:
        initializers = {
            "Psi": (shape, True),
            "nu": (shape, True),
            "tau": (latentShape + shape, True)
        }  # type: Dict[str, Tensor]
        return(initializers)

    @parameterProperty
    def mu(self) -> Tensor:
        mu = tf.zeros_like(self.Psi)
        return(mu)

    @parameterProperty
    def Psi(self) -> tf.Tensor:
        return(self.__Psi)

    @Psi.setter(name="Psi")
    def Psi(self, Psi: tf.Tensor) -> None:
        self.__Psi = Psi

    @parameterProperty
    def nu(self) -> tf.Tensor:
        return(self.__nu)

    @nu.setter(name="nu")
    def nu(self, nu: tf.Tensor) -> None:
        self.__nu = nu

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

    def cond(self) -> NnNormal:
        mu = self.mu
        Psi = self.Psi
        tau = self.tau
        mu = tf.ones_like(tau)*mu
        Psi = tf.ones_like(tau)*Psi
        name = self.name + "Cond"
        properties = Properties(name=name,
                                drawType=self.drawType,
                                updateType=self.updateType,
                                persistent=False)
        cond = NnNormal(mu=mu,
                        tau=tau/Psi,
                        properties=properties)
        return(cond)

    @property
    def shape(self) -> Tuple[int, ...]:
        return(tuple(self.mu.get_shape().as_list()))

    @property
    def latentShape(self) -> Tuple[int, ...]:
        ndims = len(self.tau.get_shape().as_list()) - len(self.shape)
        return(tuple(self.tau.get_shape().as_list()[:ndims]))
