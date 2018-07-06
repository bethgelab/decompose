from typing import Tuple, List
from tensorflow import Tensor
import tensorflow as tf

from decompose.distributions.distribution import Distribution
from decompose.likelihoods.likelihood import Likelihood


class PostU(object):

    def __init__(self, likelihood: Likelihood, prior: Distribution,
                 f: int) -> None:
        self.__likelihood = likelihood
        self.__prior = prior
        self.__f = f
        self.__K = likelihood.K

    def f(self) -> int:
        return(self.__f)

    @property
    def prior(self):
        return(self.__prior)

    def updateUf(self, Uf, Ufk, k):
        UfUpdated = tf.concat((Uf[:k], Ufk, Uf[k+1:]), 0)
        return(UfUpdated)

    def update(self, U: Tensor, X: Tensor,
               transform: bool) -> Tuple[Tensor, Tensor]:
        f, K = self.__f, self.__K

        if not transform:
            self.prior.update(data=tf.transpose(U[f]))
        else:
            self.prior.fitLatents(data=tf.transpose(U[f]))

        prepVars = self.__likelihood.prepVars(f=f, U=U, X=X)

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

        lhUfk = self.__likelihood.lhUfk(U[f], prepVars, f, k)
        postfk = lhUfk*self.prior[k].cond()
        Ufk = postfk.draw()
        Ufk = tf.expand_dims(Ufk, 0)

        normUfk = tf.norm(Ufk)
        notNanNorm = tf.logical_not(tf.is_nan(normUfk))
        finiteNorm = tf.is_finite(normUfk)
        positiveNorm = normUfk > 0.
        isValid = tf.logical_and(notNanNorm,
                                 tf.logical_and(finiteNorm,
                                                positiveNorm))
        Uf = tf.cond(isValid, lambda: self.updateUf(U[f], Ufk, k),
                     lambda: U[f])

        # TODO: if valid -> self.__likelihood.lhU()[f].updateUfk(U[f][k], k)
        Uf.set_shape(UfShape)
        U[f] = Uf
        return(U)
