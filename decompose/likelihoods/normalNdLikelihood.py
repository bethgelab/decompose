import numpy as np
from typing import Tuple, List, Dict, Any
import tensorflow as tf
from tensorflow import Tensor
import string

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.cenNormal import CenNormal
from decompose.distributions.normal import Normal
from decompose.likelihoods.likelihood import NormalLikelihood, LhU
from decompose.distributions.distribution import Properties


class NormalNdLikelihood(NormalLikelihood):

    def __init__(self, M: Tuple[int, ...], K: int=1, tau: float = 1./1e10,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 dtype=tf.float32) -> None:
        NormalLikelihood.__init__(self, M, K)
        self.__tauInit = tau
        self.__dtype = dtype
        self.__properties = Properties(name='likelihood',
                                       drawType=drawType,
                                       updateType=updateType,
                                       persistent=True)

        self.__lhU = []  # type: List[LhU]
        for f in range(self.F):
            lhUf = NormalNdLikelihoodLhU(f, self)
            self.__lhU.append(lhUf)

    def init(self) -> None:
        tau = self.__tauInit
        dtype = self.__dtype
        properties = self.__properties
        noiseDistribution = CenNormal(tau=tf.constant([tau], dtype=dtype),
                                      properties=properties)
        self.__noiseDistribution = noiseDistribution

    @property
    def noiseDistribution(self) -> CenNormal:
        return(self.__noiseDistribution)

    def residuals(self, U: List[Tensor], X: Tensor) -> Tensor:
        F = len(U)
        axisIds = string.ascii_lowercase[:F]
        subscripts = f'k{",k".join(axisIds)}->{axisIds}'
        Xhat = tf.einsum(subscripts, *U)
        residuals = X-Xhat
        return(residuals)

    def llh(self, U: List[Tensor], X: Tensor) -> Tensor:
        r = self.residuals(U, X)
        llh = tf.reduce_sum(self.noiseDistribution.llh(r))
        return(llh)

    def loss(self, U: Tuple[Tensor, ...], X: Tensor) -> Tensor:
        loss = tf.reduce_sum(self.residuals(U, X)**2)
        return(loss)

    def update(self, U: Tuple[Tensor], X: Tensor) -> None:
        if self.noiseDistribution.updateType == UpdateType.ALL:
            residuals = self.residuals(U, X)
            flattenedResiduals = tf.reshape(residuals, (-1,))[..., None]
            self.noiseDistribution.update(flattenedResiduals)

    @property
    def alpha(self) -> Tensor:
        return(self.noiseDistribution.tau[0])

    @property
    def lhU(self) -> List["LhU"]:
        return(self.__lhU)


class NormalNdLikelihoodLhU(LhU):

    def __init__(self, f: int,
                 likelihood: NormalLikelihood) -> None:
        self.__f = f
        self.__likelihood = likelihood

    def outterTensorProduct(self, Us):
        F = len(Us)
        axisIds = string.ascii_lowercase[:F]
        subscripts = f'k{",k".join(axisIds)}->{axisIds}k'
        Xhat = tf.einsum(subscripts, *Us)
        return(Xhat)

    def prepVars(self, U: List[Tensor], X: Tensor) -> Tuple[Tensor, Tensor]:
        f = self.__f
        F = len(U)

        Umf = [U[g] for g in range(F) if g != f]
        UmfOutter = self.outterTensorProduct(Umf)

        rangeFm1 = list(range(F-1))
        A = tf.tensordot(X, UmfOutter,
                         axes=([g for g in range(F) if g != f], rangeFm1))
        B = tf.tensordot(UmfOutter, UmfOutter,
                         axes=(rangeFm1, rangeFm1))
        return(A, B)

    def lhUfk(self, U: List[Tensor],
              prepVars: Tuple[Tensor, Tensor], k: Tensor) -> Distribution:
        U0 = U[self.__f]
        M = U0.get_shape().as_list()[1]
        alpha = self.__likelihood.alpha
        assertAlphaIs0 = tf.Assert(tf.reduce_all(tf.logical_not(tf.equal(alpha, 0.))), [alpha], name='LHalphaIs0')
        assertAlphaNotPos = tf.Assert(tf.reduce_all(tf.greater(alpha, 0.)), [alpha], name='LHalphaNotPositive')
        with tf.control_dependencies([assertAlphaIs0, assertAlphaNotPos]):
            alpha = alpha + 0.

        A, B = prepVars
        Xv = tf.reshape(tf.slice(A, [0, k], [M, 1]), [-1])
        Bk = tf.slice(B, [0, k], [B.get_shape().as_list()[0], 1])
        Bkk = tf.reshape(tf.slice(B, [k, k], [1, 1]), [-1])

        Uk = tf.slice(U0, [k, 0], [1, M])

        UVTv = tf.reshape(tf.matmul(tf.transpose(Bk), U0), [-1])
        uvTv = tf.reshape(Uk*Bkk, [-1])

        mlMeanPrecisionDivAlpha = Xv - UVTv + uvTv
        Bkk = tf.where(tf.equal(Bkk, 0.), tf.ones_like(Bkk)*np.finfo(np.float32).eps, Bkk)
        mlPrecisionDivAlpha = Bkk*tf.ones_like(M, dtype=U0.dtype)
        mlMean = mlMeanPrecisionDivAlpha/mlPrecisionDivAlpha

        mlPrecision = tf.multiply(mlPrecisionDivAlpha, alpha)
        mlPrecision = tf.where(tf.equal(mlPrecision, 0.), tf.ones_like(mlPrecision)*np.finfo(np.float32).eps, mlPrecision)

        tau = mlPrecision
        assertTauIs0 = tf.Assert(tf.reduce_all(tf.logical_not(tf.equal(tau, 0.))), [tau], name='LHtauIs0')
        assertTauNotPos = tf.Assert(tf.reduce_all(tf.greater(tau, 0.)), [tau], name='LHtauNotPositive')
        with tf.control_dependencies([assertTauIs0, assertTauNotPos]):
            tau = tau + 0.

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
