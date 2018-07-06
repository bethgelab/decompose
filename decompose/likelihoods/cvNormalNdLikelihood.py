import numpy as np
from typing import Tuple, List
import tensorflow as tf
from tensorflow import Tensor
import string

from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.cenNormal import CenNormal
from decompose.likelihoods.likelihood import Likelihood
from decompose.distributions.distribution import Properties
from decompose.cv.cv import CV


class CVNormalNdLikelihood(Likelihood):

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
        F = len(U)
        axisIds = string.ascii_lowercase[:F]
        subscripts = f'k{",k".join(axisIds)}->{axisIds}'
        Xhat = tf.einsum(subscripts, *U)
        residuals = tf.reshape(X-Xhat, (-1,))
        indices = tf.cast(tf.where(tf.reshape(self.testMask, (-1,))),
                          dtype=tf.int32)
        testResiduals = tf.gather_nd(residuals, indices)
        return(testResiduals)

    def trainResiduals(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
        F = len(U)
        axisIds = string.ascii_lowercase[:F]
        subscripts = f'k{",k".join(axisIds)}->{axisIds}'
        Xhat = tf.einsum(subscripts, *U)
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

    def outterTensorProduct(self, Us):
        F = len(Us)
        axisIds = string.ascii_lowercase[:F]
        subscripts = f'k{",k".join(axisIds)}->{axisIds}k'
        Xhat = tf.einsum(subscripts, *Us)
        return(Xhat)

    def calcB(self, mask, UmfOutter, f, F):
        axisIds0 = (string.ascii_lowercase[:f]
                    + "x"
                    + string.ascii_lowercase[f:F-1])
        axisIds1 = string.ascii_lowercase[:F-1] + "y"
        axisIds2 = string.ascii_lowercase[:F-1] + "z"
        subscripts = (axisIds0 + ","
                      + axisIds1 + ","
                      + axisIds2 + "->"
                      + "xyz")
        B = tf.einsum(subscripts, mask, UmfOutter, UmfOutter)
        return(B)

    def prepVars(self, f: int, U: List[Tensor],
                 X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mask = tf.cast(self.trainMask, dtype=U[0].dtype)
        F = self.F

        Umf = [U[g] for g in range(F) if g != f]
        UmfOutter = self.outterTensorProduct(Umf)

        rangeFm1 = list(range(F-1))
        A = tf.tensordot(X*mask, UmfOutter,
                         axes=([g for g in range(F) if g != f], rangeFm1))

        B = self.calcB(mask, UmfOutter, f, F)
        alpha = self.noiseDistribution.tau
        return(A, B, alpha)
