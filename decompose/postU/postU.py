from typing import Tuple, List
import numpy as np
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
        self.__K = likelihood.K
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
            self.prior.update(data=tf.transpose(U[f]))

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
        f = self.__f

        UfShape = U[f].get_shape()

        lhUfk = self.__likelihood.lhU[f].lhUfk(U, prepVars, k)
        postfk = lhUfk*self.prior[k].cond()
        Ufk = postfk.draw()
        Ufk = tf.expand_dims(Ufk, 0)

        allZero = tf.reduce_all(tf.equal(Ufk, 0.))
        isFinite = tf.reduce_all(tf.is_finite(Ufk))
        isValid = tf.logical_and(isFinite, tf.logical_not(allZero))
        Uf = tf.cond(isValid, lambda: self.updateUf(U[f], Ufk, k),
                     lambda: U[f])

        # TODO: if valid -> self.__likelihood.lhU()[f].updateUfk(U[f][k], k)
        Uf.set_shape(UfShape)
        U[f] = Uf
        return(U)
