from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Dict, Any
from numpy import ndarray
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.distribution import Distribution


class LhU(metaclass=ABCMeta):

    @abstractmethod
    def prepVars(self, U: List[Tensor], X: Tensor) -> Tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def lhUfk(self, U: List[Tensor],
              prepVars: Tuple[Tensor, Tensor], k: Tensor) -> Distribution:
        ...

    @abstractmethod
    def newUfk(self, Ufk: Tensor, k: Tensor) -> None:
        ...


class Likelihood(metaclass=ABCMeta):

    def __init__(self, M: Tuple[int, ...], K: int) -> None:
        self.__K = K
        self.__M = M
        self.__F = len(M)

    @abstractmethod
    def update(self, U: List[Tensor], X: Tensor) -> None:
        ...

    @property
    @abstractmethod
    def lhU(self) -> List["LhU"]:
        ...

    @property
    @abstractmethod
    def noiseDistribution(self) -> Distribution:
        ...

    @abstractmethod
    def residuals(self, U: List[Tensor], X: Tensor) -> tf.Tensor:
        ...

    @abstractmethod
    def init(self) -> None:
        ...

    @property
    def M(self) -> Tuple[int, ...]:
        return(self.__M)

    @property
    def F(self) -> int:
        return(self.__F)

    @property
    def K(self) -> int:
        return(self.__K)

    @property
    def id(self) -> str:
        likelihoodName = self.__class__.__name__
        K = str(self.K)
        id = likelihoodName + K
        return(id)


class NormalLikelihood(Likelihood):
    @abstractmethod
    def alpha(self) -> Tensor:
        ...
