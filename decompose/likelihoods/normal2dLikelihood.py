from typing import Tuple, List
import tensorflow as tf
from tensorflow import Tensor, DType

from decompose.distributions.distribution import Properties
from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.cenNormal import CenNormal
from decompose.likelihoods.likelihood import Likelihood


class Normal2dLikelihood(Likelihood):

    def __init__(self, M: Tuple[int, ...], K: int=1, tau: float = 1./1e10,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 dtype: DType = tf.float32) -> None:
        Likelihood.__init__(self, M, K)
        self.__tauInit = tau
        self.__dtype = dtype
        self.__properties = Properties(name='likelihood',
                                       drawType=drawType,
                                       dtype=dtype,
                                       updateType=updateType,
                                       persistent=True)

    def init(self, data: Tensor) -> None:
        tau = self.__tauInit
        dtype = self.__dtype
        properties = self.__properties
        noiseDistribution = CenNormal(tau=tf.constant([tau], dtype=dtype),
                                      properties=properties)
        self.__noiseDistribution = noiseDistribution

    @property
    def noiseDistribution(self) -> CenNormal:
        return(self.__noiseDistribution)

    def residuals(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
        assert(len(U) == 2)
        U0, U1 = U
        Xhat = tf.matmul(tf.transpose(U0), U1)
        residuals = tf.reshape(X-Xhat, (-1,))
        return(residuals)

    def llh(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
        r = self.residuals(U, X)
        llh = tf.reduce_sum(self.noiseDistribution.llh(r))
        return(llh)

    def loss(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
        loss = tf.reduce_sum(self.residuals(U, X)**2)
        return(loss)

    def update(self, U: Tuple[Tensor, ...], X: Tensor) -> None:
        if self.noiseDistribution.updateType == UpdateType.ALL:
            residuals = self.residuals(U, X)
            flattenedResiduals = tf.reshape(residuals, (-1,))[..., None]
            self.noiseDistribution.update(flattenedResiduals)

    def prepVars(self, f: int, U: List[Tensor],
                 X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if f == 0:
            U1 = U[1]
        else:
            U1 = U[0]
            X = tf.transpose(X)

        U1T = tf.transpose(U1)
        A = tf.matmul(X, U1T)
        B = tf.matmul(U1, U1T)
        alpha = self.noiseDistribution.tau

        return(A, B, alpha)
