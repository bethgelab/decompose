from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.distribution import Distribution
from decompose.distributions.nnNormal import NnNormal
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.cenNnTAlgorithms import CenNnTAlgorithms


class CenNnT(Distribution):
    def __init__(self,
                 Psi: Tensor = tf.constant([1.]),
                 nu: Tensor = tf.constant([1.]),
                 tau: Tensor = tf.constant([1.]),
                 name: str = "NA",
                 algorithms: Type[Algorithms] = CenNnTAlgorithms,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 persistent: bool = True) -> None:
        Distribution.__init__(self,
                              shape=Psi.shape,
                              latentShape=(),
                              name=name,
                              drawType=drawType,
                              dtype=Psi.dtype,
                              updateType=updateType,
                              persistent=persistent,
                              algorithms=algorithms)
        self._init({"Psi": Psi,
                    "nu": nu,
                    "tau": tau})

    @staticmethod
    def initializers(shape: Tuple[int, ...] = (1,),
                     latentShape: Tuple[int, ...] = (1000,),
                     dtype: np.dtype = np.float32) -> Dict[str, Tensor]:

        dtype = tf.as_dtype(dtype)
        one = tf.constant(1., dtype=dtype)
        exponential = tf.distributions.Exponential(rate=one)
        initializers = {
            "Psi": exponential.sample(sample_shape=shape),
            "nu": exponential.sample(sample_shape=shape),
            "tau": exponential.sample(sample_shape=latentShape + shape)
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
    @classmethod
    def nonNegative(cls) -> bool:
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
        cond = NnNormal(mu=mu, tau=tau/Psi, name=name,
                        drawType=self.drawType,
                        updateType=self.updateType,
                        persistent=False)
        return(cond)
