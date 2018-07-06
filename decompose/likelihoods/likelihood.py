from abc import ABCMeta, abstractmethod
from typing import Tuple, List
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.normal import Normal
from decompose.distributions.distribution import Distribution, Properties


class Likelihood(metaclass=ABCMeta):

    def __init__(self, M: Tuple[int, ...], K: int) -> None:
        self.__K = K
        self.__M = M
        self.__F = len(M)

    @abstractmethod
    def prepVars(self, f: int, U: List[Tensor],
                 X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        ...

    def lhUfk(self, Uf: Tensor, prepVars: Tuple[Tensor, ...],
              f: int, k: Tensor) -> Distribution:
        XVT, VVT, alpha = prepVars
        XvT = XVT[:, k]
        VvT = VVT[..., k]
        vvT = VVT[..., k, k]
        Ufk = Uf[k]

        UVvT = tf.reduce_sum(tf.transpose(Uf)*VvT, axis=-1)
        uvvT = Ufk*vvT
        Xtildev = XvT - UVvT + uvvT

        mu = Xtildev/vvT
        tau = vvT*alpha

        properties = Properties(name=f"lhU{f}k",
                                drawType=self.noiseDistribution.drawType,
                                updateType=self.noiseDistribution.updateType,
                                persistent=False)
        lhUfk = Normal(mu=mu, tau=tau, properties=properties)
        return(lhUfk)

    @abstractmethod
    def update(self, U: Tuple[Tensor, ...], X: Tensor) -> None:
        ...

    @property
    @abstractmethod
    def noiseDistribution(self) -> Distribution:
        ...

    @abstractmethod
    def init(self, data: Tensor) -> None:
        ...

    @abstractmethod
    def llh(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
        ...

    @abstractmethod
    def loss(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
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
