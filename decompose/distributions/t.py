from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.distribution import Distribution
from decompose.distributions.normal import Normal
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.tAlgorithms import TAlgorithms


class T(Distribution):
    def __init__(self,
                 mu: Tensor = tf.constant([0.]),
                 Psi: Tensor = tf.constant([1.]),
                 nu: Tensor = tf.constant([1.]),
                 tau: Tensor = tf.constant([1.]),
                 name: str = "NA",
                 algorithms: Type[Algorithms] = TAlgorithms,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 persistent: bool = True) -> None:
        Distribution.__init__(self,
                              shape=mu.shape,
                              latentShape=(),
                              name=name,
                              drawType=drawType,
                              dtype=mu.dtype,
                              updateType=updateType,
                              persistent=persistent,
                              algorithms=algorithms)
        self._init({"mu": mu,
                    "Psi": Psi,
                    "nu": nu,
                    "tau": tau})

    @staticmethod
    def initializers(shape: Tuple[int, ...] = (1,),
                     latentShape: Tuple[int, ...] = (1000,),
                     dtype: np.dtype = np.float32) -> Dict[str, Tensor]:
        initializers = {
            "mu": tf.constant(np.random.normal(size=shape), dtype=dtype),
            "Psi": tf.constant(np.random.exponential(size=shape), dtype=dtype),
            "nu": tf.constant(np.random.exponential(size=shape), dtype=dtype),
            "tau": tf.constant(np.random.exponential(size=latentShape + shape),
                               dtype=dtype)
        }  # type: Dict[str, Tensor]
        return(initializers)

    @parameterProperty
    def mu(self) -> Tensor:
        return(self.__mu)

    @mu.setter(name="mu")
    def mu(self, mu: Tensor) -> None:
        self.__mu = mu

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
    @classmethod
    def nonNegative(cls) -> bool:
        return(False)

    @property
    def homogenous(self) -> bool:
        return(False)

    def cond(self) -> Normal:
        mu = self.mu
        Psi = self.Psi
        tau = self.tau
        mu = tf.ones_like(tau)*mu
        Psi = tf.ones_like(tau)*Psi
        name = self.name + "Cond"
        cond = Normal(mu=mu, tau=tau/Psi, name=name,
                      drawType=self.drawType,
                      updateType=self.updateType,
                      persistent=False)
        return(cond)
