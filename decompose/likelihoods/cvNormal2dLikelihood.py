import numpy as np
from typing import Tuple, List, Dict, Any
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.cenNormal import CenNormal
from decompose.distributions.normal import Normal
from decompose.likelihoods.likelihood import NormalLikelihood, LhU
from decompose.distributions.distribution import Properties


class CVNormal2dLikelihood(NormalLikelihood):

    def __init__(self, M: Tuple[int, ...], K: int=1, tau: float = 1./1e10,
                 trainsetProb: float = 0.8,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 dtype=tf.float32) -> None:
        NormalLikelihood.__init__(self, M, K)
        self.__trainsetProb = trainsetProb
        self.__tauInit = tau
        self.__dtype = dtype
        self.__properties = Properties(name='likelihood',
                                       drawType=drawType,
                                       updateType=updateType,
                                       persistent=True)

        self.__lhU = []  # type: List[LhU]
        for f in range(self.F):
            lhUf = Normal2dLikelihoodLhU(f, self)
            self.__lhU.append(lhUf)

    @staticmethod
    def type():
        return(CVNormal2dLikelihood)

    def init(self, data: Tensor) -> None:
        tau = self.__tauInit
        dtype = self.__dtype
        properties = self.__properties
        noiseDistribution = CenNormal(tau=tf.constant([tau], dtype=dtype),
                                      properties=properties)
        self.__noiseDistribution = noiseDistribution

        observedMask = tf.logical_not(tf.is_nan(data))
        trainsetProb = self.__trainsetProb
        r = tf.distributions.Uniform().sample(sample_shape=self.M)
        trainMask = tf.less(r, trainsetProb)
        trainMask = tf.get_variable("trainMask",
                                    dtype=trainMask.dtype,
                                    initializer=trainMask)
        trainMask = tf.logical_and(trainMask, observedMask)
        testMask = tf.logical_and(observedMask,
                                  tf.logical_not(trainMask))
        self.__observedMask = observedMask
        self.__trainMask = trainMask
        self.__testMask = testMask

    @property
    def observedMask(self) -> Tensor:
        return(self.__observedMask)

    @property
    def trainMask(self) -> Tensor:
        return(self.__trainMask)

    @property
    def testMask(self) -> Tensor:
        return(self.__testMask)

    @property
    def noiseDistribution(self) -> CenNormal:
        return(self.__noiseDistribution)

    def residuals(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
        return(self.testResiduals(U, X))

    def testResiduals(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
        assert(len(U) == 2)
        U0, U1 = U
        Xhat = tf.matmul(tf.transpose(U0), U1)
        residuals = tf.reshape(X-Xhat, (-1,))
        indices = tf.cast(tf.where(tf.reshape(self.testMask, (-1,))),
                          dtype=tf.int32)
        testResiduals = tf.gather_nd(residuals, indices)
        return(testResiduals)

    def trainResiduals(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
        assert(len(U) == 2)
        U0, U1 = U
        Xhat = tf.matmul(tf.transpose(U0), U1)
        residuals = tf.reshape(X-Xhat, (-1,))
        indices = tf.cast(tf.where(tf.reshape(self.trainMask, (-1,))),
                          dtype=tf.int32)
        trainResiduals = tf.gather_nd(residuals, indices)
        return(trainResiduals)

    def llh(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
        testsetProb = 1. - self.__trainsetProb
        r = self.testResiduals(U, X)
        llh = tf.reduce_sum(self.noiseDistribution.llh(r))/testsetProb
        return(llh)

    def loss(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
        loss = tf.reduce_sum(self.testResiduals(U, X)**2)
        return(loss)

    def update(self, U: Tuple[Tensor, ...], X: Tensor) -> None:
        if self.noiseDistribution.updateType == UpdateType.ALL:
            residuals = self.trainResiduals(U, X)
            flattenedResiduals = residuals[..., None]
            self.noiseDistribution.update(flattenedResiduals)

    @property
    def alpha(self) -> Tensor:
        return(self.noiseDistribution.tau[0])

    @property
    def lhU(self) -> List["LhU"]:
        return(self.__lhU)


class Normal2dLikelihoodLhU(LhU):

    def __init__(self, f: int,
                 likelihood: NormalLikelihood) -> None:
        self.__f = f
        self.__g = (self.__f-1)**2
        self.__likelihood = likelihood

    def prepVars(self, U: List[Tensor], X: Tensor) -> Tuple[Tensor, Tensor]:
        U1 = self.__likelihood.lhU[self.__g].getUfRep(U[self.__g])
        U1T = tf.transpose(U1, [1, 0])
        X = tf.transpose(X, [self.__f, self.__g])
        trainMask = tf.cast(self.__likelihood.trainMask, dtype=U[0].dtype)
        mask = tf.transpose(trainMask, [self.__f, self.__g])

        A = tf.matmul(X*mask, U1T)
        B = tf.einsum("mn,in,jn->mij", mask, U1, U1)
        return(A, B)

    def lhUfk(self, U: List[Tensor],
              prepVars: Tuple[Tensor, ...], k: Tensor) -> Distribution:
        U0 = U[self.__f]
        K, M = U0.get_shape().as_list()
        alpha = self.__likelihood.alpha

        A, B = prepVars
        Xv = tf.slice(A, [0, k], [M, 1])[..., 0]
        Bk = tf.slice(B, [0, 0, k], [M, K, 1])[..., 0]
        Bkk = tf.slice(B, [0, k, k], [M, 1, 1])[:, 0, 0]
        Uk = tf.slice(U0, [k, 0], [1, M])[0]
        UVTv = tf.reduce_sum(tf.transpose(U0)*Bk, axis=1)
        uvTv = Uk*Bkk

        mlMeanPrecisionDivAlpha = Xv - UVTv + uvTv
        mlPrecisionDivAlpha = Bkk
        mlMean = mlMeanPrecisionDivAlpha/mlPrecisionDivAlpha
        mlPrecision = tf.multiply(mlPrecisionDivAlpha, alpha)

        noiseDistribution = self.__likelihood.noiseDistribution
        properties = Properties(name="mlU{}".format(self.__f),
                                drawType=noiseDistribution.drawType,
                                updateType=noiseDistribution.updateType,
                                persistent=False)

        lhUfk = Normal(mu=mlMean,
                       tau=mlPrecision,
                       properties=properties)
        return(lhUfk)

    def newUfk(self, Ufk: Tensor, k: Tensor) -> None:
        pass

    def rescaleUfk(self, c: Tensor) -> None:
        pass

    def getUfRep(self, Uf: Tensor) -> Tensor:
        return(Uf)
