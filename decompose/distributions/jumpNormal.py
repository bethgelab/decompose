from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.jumpNormalAlgorithms import JumpNormalAlgorithms


class JumpNormal(Distribution):
    def __init__(self,
                 mu: Tensor = tf.constant([0.]),
                 tau: Tensor = tf.constant([1.]),
                 nu: Tensor = tf.constant([0.]),
                 beta: Tensor = tf.constant([1.]),
                 name: str = "NA",
                 algorithms: Type[Algorithms] = JumpNormalAlgorithms,
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
                    "tau": tau,
                    "nu": nu,
                    "beta": beta})

    @staticmethod
    def initializers(shape: Tuple[int, ...] = (1,),
                     latentShape: Tuple[int, ...] = (),
                     dtype: np.dtype = np.float32) -> Dict[str, Tensor]:
        initializers = {
            "mu": tf.constant(np.random.normal(size=shape), dtype=dtype),
            "tau": tf.constant(np.random.exponential(size=shape), dtype=dtype),
            "nu": tf.constant(np.random.normal(size=shape), dtype=dtype),
            "beta": tf.constant(np.random.exponential(size=shape), dtype=dtype)

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
    @classmethod
    def nonNegative(cls) -> bool:
        return(False)

    @property
    def homogenous(self) -> bool:
        return(False)

    def cond(self) -> Distribution:
        return(self)
