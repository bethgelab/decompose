import numpy as np
from typing import Tuple, List
import tensorflow as tf
from tensorflow import Tensor
import string

from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.cenNormal import CenNormal
from decompose.likelihoods.likelihood import Likelihood
from decompose.distributions.distribution import Properties


class NormalNdLikelihood(Likelihood):

    def __init__(self, M: Tuple[int, ...], K: int=1, tau: float = 1./1e10,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 dtype=tf.float32) -> None:
        Likelihood.__init__(self, M, K)
        self.__tauInit = tau
        self.__dtype = dtype
        self.__properties = Properties(name='likelihood',
                                       drawType=drawType,
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
        F = len(U)
        axisIds = string.ascii_lowercase[:F]
        subscripts = f'k{",k".join(axisIds)}->{axisIds}'
        Xhat = tf.einsum(subscripts, *U)
        residuals = X-Xhat
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

    def outterTensorProduct(self, Us):
        F = len(Us)
        axisIds = string.ascii_lowercase[:F]
        subscripts = f'k{",k".join(axisIds)}->{axisIds}k'
        Xhat = tf.einsum(subscripts, *Us)
        return(Xhat)

    def prepVars(self, f: int, U: List[Tensor],
                 X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        F = self.F
        Umf = [U[g] for g in range(F) if g != f]
        UmfOutter = self.outterTensorProduct(Umf)

        rangeFm1 = list(range(F-1))
        A = tf.tensordot(X, UmfOutter,
                         axes=([g for g in range(F) if g != f], rangeFm1))
        B = tf.tensordot(UmfOutter, UmfOutter,
                         axes=(rangeFm1, rangeFm1))
        alpha = self.noiseDistribution.tau
        return(A, B, alpha)
