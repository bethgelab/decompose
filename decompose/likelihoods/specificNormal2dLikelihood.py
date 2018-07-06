import numpy as np
from typing import Tuple, List
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.distribution import Properties
from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.cenNormal import CenNormal
from decompose.likelihoods.likelihood import Likelihood


class SpecificNormal2dLikelihood(Likelihood):

    def __init__(self, M: Tuple[int, ...], K: int=1, tau: float = 1./1e10,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 dtype=tf.float32) -> None:
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
        properties = self.__properties
        tau = tf.ones_like(data[0])*tau  # TODO is using ones really useful
        noiseDistribution = CenNormal(tau=tau,
                                      properties=properties)
        self.__noiseDistribution = noiseDistribution

    @property
    def noiseDistribution(self) -> CenNormal:
        return(self.__noiseDistribution)

    def residuals(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
        assert(len(U) == 2)
        U0, U1 = U
        Xhat = tf.matmul(tf.transpose(U0), U1)
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
            self.noiseDistribution.update(residuals)

    def prepVars(self, f: int, U: List[Tensor],
                 X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if f == 0:
            U1 = U[1]
            alpha1 = self.noiseDistribution.tau
            alpha = tf.ones_like(X[:, 0])
        elif f == 1:
            U1 = U[0]
            alpha1 = tf.ones_like(X[:, 0])
            alpha = self.noiseDistribution.tau
            X = tf.transpose(X)

        U1T = tf.transpose(U1)
        A = tf.matmul(X, U1T*alpha1[..., None])
        B = tf.matmul(U1*alpha1, U1T)
        return(A, B, alpha)
