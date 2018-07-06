import numpy as np
from typing import Tuple, List
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.cenNormal import CenNormal
from decompose.likelihoods.likelihood import Likelihood
from decompose.distributions.distribution import Properties
from decompose.cv.cv import CV


class CVNormal2dLikelihood(Likelihood):

    def __init__(self, M: Tuple[int, ...], K: int=1, tau: float = 1./1e10,
                 cv: CV = None,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 dtype=tf.float32) -> None:
        Likelihood.__init__(self, M, K)
        self.__cv = cv
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
        observedMask = tf.logical_not(tf.is_nan(data))
        trainMask = tf.logical_not(self.cv.mask(X=data))
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
    def cv(self) -> CV:
        return(self.__cv)

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
        r = self.testResiduals(U, X)
        llh = tf.reduce_sum(self.noiseDistribution.llh(r))
        return(llh)

    def loss(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
        loss = tf.reduce_sum(self.testResiduals(U, X)**2)
        return(loss)

    def update(self, U: Tuple[Tensor, ...], X: Tensor) -> None:
        if self.noiseDistribution.updateType == UpdateType.ALL:
            residuals = self.trainResiduals(U, X)
            flattenedResiduals = residuals[..., None]
            self.noiseDistribution.update(flattenedResiduals)

    def prepVars(self, f: int, U: List[Tensor],
                 X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        trainMask = tf.cast(self.__trainMask, dtype=U[0].dtype)
        if f == 0:
            U1 = U[1]
        else:
            U1 = U[0]
            X = tf.transpose(X)
            trainMask = tf.transpose(trainMask)
        U1T = tf.transpose(U1)

        A = tf.matmul(X*trainMask, U1T)
        B = tf.einsum("mn,in,jn->mij", trainMask, U1, U1)
        alpha = self.noiseDistribution.tau
        return(A, B, alpha)
