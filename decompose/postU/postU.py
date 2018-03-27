from typing import Tuple, List
import numpy as np
from numpy import ndarray
from tensorflow import Tensor
import tensorflow as tf

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import DrawType
from decompose.likelihoods.likelihood import Likelihood


class PostU(object):

    def __init__(self, likelihood: Likelihood, prior: Distribution,
                 f: int, normalize: bool = False) -> None:
        self.__likelihood = likelihood
        self.__prior = prior
        self.__f = f
        self.__K = prior.shape[0]
        self.__normalize = normalize

    def f(self) -> int:
        return(self.__f)

    @property
    def prior(self):
        return(self.__prior)

    def updateUf(self, Uf, Ufk, k):
        UfUpdated = tf.concat((Uf[:k], Ufk, Uf[k+1:]), 0)
        return(UfUpdated)

    def update(self, U: Tensor, X: Tensor, t) -> Tuple[Tensor, Tensor]:
        f, K = self.__f, self.__K

        if not t:
            self.prior.update(data=U[f])

        prepVars = self.__likelihood.lhU[f].prepVars(U, X)

        def cond(k, Uf):
            return(tf.less(k, K))

        def body(k, U):
            U = self.updateK(k, prepVars, U)
            return(k+1, U)

        k = tf.constant(0)
        loop_vars = [k, list(U)]

        _, U = tf.while_loop(cond, body, loop_vars)
        return(U[f])

    def updateK(self, k, prepVars, U):
        if self.prior.drawType == DrawType.SAMPLE:
            return(self.updateKSample(k, prepVars, U))
        else:
            return(self.updateKSample(k, prepVars, U))

    def updateKSample(self, k, prepVars, U):
        f = self.__f

        UfShape = U[f].get_shape()

        lhUfk = self.__likelihood.lhU[f].lhUfk(U, prepVars, k)
        postfk = lhUfk*self.prior[k].cond()
        Ufk = postfk.draw()

        Ufk = tf.expand_dims(Ufk, 0)

        isValid = tf.reduce_all(tf.is_finite(Ufk))
        Uf = tf.cond(isValid, lambda: self.updateUf(U[f], Ufk, k),
                     lambda: U[f])

        # TODO: if valid -> self.__likelihood.lhU()[f].updateUfk(U[f][k], k)
        Uf.set_shape(UfShape)
        U[f] = Uf
        return(U)

    def updateKMode(self, k, prepVars, U, normalize, absMax):
        f = self.__f

        UfShape = U[f].get_shape()

        lhUfk = self.__likelihood.lhU[f].lhUfk(U, prepVars, k)
        postfk = lhUfk*self.prior[k].cond()
        Ufk = postfk.draw()
        if normalize:
            Ufk = Ufk/tf.norm(Ufk)
        Ufk = tf.expand_dims(Ufk, 0)

        Ufk = tf.where(tf.greater(tf.abs(Ufk), absMax[k]),
                       tf.sign(Ufk)*absMax[k], Ufk)

        isValid = tf.reduce_all(tf.is_finite(Ufk))
        Uf = tf.cond(isValid, lambda: self.updateUf(U[f], Ufk, k),
                     lambda: U[f])

        # TODO: if valid -> self.__likelihood.lhU()[f].updateUfk(U[f][k], k)
        Uf.set_shape(UfShape)
        U[f] = Uf

        return(U)
