import numpy as np
from typing import Tuple, List, Dict, Any
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.distribution import Distribution, Properties
from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.cenNormalRankOne import CenNormalRankOne
from decompose.distributions.normal import Normal
from decompose.likelihoods.likelihood import NormalLikelihood, LhU


class AllSpecificNormal2dLikelihood(NormalLikelihood):

    def __init__(self, M: Tuple[int, ...], K: int=1, tau: float = 1./1e2,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 dtype=tf.float32) -> None:
        NormalLikelihood.__init__(self, M, K)
        self.__tauInit = tau
        self.__dtype = dtype
        self.__properties = Properties(name='likelihood',
                                       drawType=drawType,
                                       dtype=dtype,
                                       updateType=updateType,
                                       persistent=True)

        self.__lhU = []  # type: List[LhU]
        for f in range(self.F):
            lhUf = Normal2dLikelihoodLhU(f, self)
            self.__lhU.append(lhUf)

    def init(self, data: Tensor) -> None:
        M, N = data.get_shape().as_list()
        tau0Init = tf.random_uniform(shape=(M,))
        tau1Init = tf.random_uniform(shape=(N,))
        properties = self.__properties
        noiseDistribution = CenNormalRankOne(tau0=tau0Init,
                                             tau1=tau1Init,
                                             properties=properties)
        self.__noiseDistribution = noiseDistribution

    @property
    def noiseDistribution(self) -> CenNormalRankOne:
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

    @property
    def alpha(self) -> Tensor:
        return(self.noiseDistribution.tau0, self.noiseDistribution.tau1)

    @property
    def lhU(self) -> List["LhU"]:
        return(self.__lhU)


class Normal2dLikelihoodLhU(LhU):

    def __init__(self, f: int,
                 likelihood: NormalLikelihood) -> None:
        self.__f = f
        self.__g = (self.__f-1)**2
        self.__likelihood = likelihood

    def prepVars(self, U: List[Tensor], X: Tensor) -> Tuple[Tensor, ...]:
        if self.__f == 0:
            U1 = self.__likelihood.lhU[1].getUfRep(U[1])
            alpha0, alpha1 = self.__likelihood.alpha
        elif self.__f == 1:
            U1 = self.__likelihood.lhU[0].getUfRep(U[0])
            alpha1, alpha0 = self.__likelihood.alpha
            X = tf.transpose(X)

        norms = tf.norm(U1, axis=1)
        U1n = U1/norms[..., None]
        U1Tn = tf.transpose(U1n)
        A = tf.matmul(X*alpha0[..., None]*alpha1, U1Tn)
        B = tf.matmul(U1n*alpha1, U1Tn)[None]
        B = B * alpha0[..., None, None]
        return(A, B, norms)

    def prepVarsAdd(self, U: List[Tensor], X: Tensor) -> Tuple[Tensor, ...]:
        if self.__f == 0:
            U1 = self.__likelihood.lhU[1].getUfRep(U[1])
            alpha1, alpha0 = self.__likelihood.alpha
        elif self.__f == 1:
            U1 = self.__likelihood.lhU[0].getUfRep(U[0])
            alpha0, alpha1 = self.__likelihood.alpha
            X = tf.transpose(X)

        norms = tf.norm(U1, axis=1)
        U1n = U1/norms[..., None]
        U1Tn = tf.transpose(U1n)
        A = tf.matmul(X*alpha0 + X*alpha1[:, None], U1Tn)
        B0 = tf.matmul(U1n*alpha0, U1Tn)[None]
        B1 = tf.matmul(U1n, U1Tn)[None] * alpha1[..., None, None]
        B = B0 + B1
        return(A, B, norms)

    def lhUfk(self, U: List[Tensor],
              prepVars: Tuple[Tensor, ...], k: Tensor) -> Distribution:
        A, B, norms = prepVars
        U0 = U[self.__f]*norms[..., None]
        M = U0.get_shape().as_list()[1]
        K = B.get_shape().as_list()[1]

        Xv = tf.slice(A, [0, k], [M, 1])[..., 0]
        Bk = tf.slice(B, [0, 0, k], [M, K, 1])[..., 0]
        Bkk = tf.slice(B, [0, k, k], [M, 1, 1])[..., 0, 0]
        vTv = Bkk

        Uk = tf.slice(U0, [k, 0], [1, M])[0]
        UVTv = tf.reduce_sum(tf.transpose(Bk) * U0, axis=0)
        uvTv = Uk*Bkk

        mlMeanPrecisionDivAlpha = Xv - UVTv + uvTv
        mlPrecisionDivAlpha = vTv
        mlMean = mlMeanPrecisionDivAlpha/mlPrecisionDivAlpha
        mlPrecision = mlPrecisionDivAlpha
        tau = mlPrecision

        noiseDistribution = self.__likelihood.noiseDistribution
        properties = Properties(name="lhU{}".format(self.__f),
                                drawType=noiseDistribution.drawType,
                                updateType=noiseDistribution.updateType,
                                persistent=False)
        lhUfk = Normal(mu=mlMean,
                       tau=tau,
                       properties=properties)
        return(lhUfk)

    def newUfk(self, Ufk: Tensor, k: Tensor) -> None:
        pass

    def rescaleUfk(self, c: Tensor) -> None:
        pass

    def getUfRep(self, Uf: Tensor) -> Tensor:
        return(Uf)
