from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.distribution import Distribution
from decompose.distributions.cenLaplace import CenLaplace
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.cenDoubleLomaxAlgorithms import CenDoubleLomaxAlgorithms


class CenDoubleLomax(Distribution):
    def __init__(self,
                 alpha: Tensor = tf.constant([1.]),
                 beta: Tensor = tf.constant([1.]),
                 tau: Tensor = tf.constant([1.]),
                 name: str = "NA",
                 algorithms: Type[Algorithms] = CenDoubleLomaxAlgorithms,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 persistent: bool = True) -> None:
        Distribution.__init__(self,
                              shape=alpha.shape,
                              latentShape=(),
                              name=name,
                              drawType=drawType,
                              dtype=alpha.dtype,
                              updateType=updateType,
                              persistent=persistent,
                              algorithms=algorithms)
        self._init({"alpha": alpha,
                    "beta": beta,
                    "tau": tau})

    @staticmethod
    def initializers(shape: Tuple[int, ...] = (1,),
                     latentShape: Tuple[int, ...] = (1000,),
                     dtype: np.dtype = np.float32) -> Dict[str, Tensor]:
        initializers = {
            "alpha": tf.constant(np.random.exponential(size=shape), dtype=dtype),
            "beta": tf.constant(np.random.exponential(size=shape), dtype=dtype),
            "tau": tf.constant(np.random.exponential(size=latentShape + shape),
                               dtype=dtype)
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
    @classmethod
    def nonNegative(cls) -> bool:
        return(True)

    @property
    def homogenous(self) -> bool:
        return(False)

    def cond(self) -> CenLaplace:
        tau = self.tau
        name = self.name + "Cond"
        cond = CenLaplace(beta=1./tau,
                          name=name,
                          drawType=self.drawType,
                          updateType=self.updateType,
                          persistent=False)
        return(cond)
