import numpy as np
import tensorflow as tf

from decompose.distributions.distribution import Distribution
from decompose.distributions.normal import Normal
from decompose.distributions.nnNormal import NnNormal
from decompose.distributions.product import Product


class NormalNnNormal(Product):

    def fromUnordered(self, d0: Distribution,
                      d1: Distribution) -> NnNormal:
        if isinstance(d0, Normal) and isinstance(d1, NnNormal):
            return(self.product(d0, d1))
        elif isinstance(d1, Normal) and isinstance(d0, NnNormal):
            return(self.product(d1, d0))
        else:
            raise ValueError("Expecting Normal and NnNormal")

    def product(self, n: Normal, nnn: NnNormal) -> NnNormal:
        mu = self.mu(n, nnn)
        tau = self.tau(n, nnn)
        otherParams = self.productParams(n, nnn)
        return(NnNormal(mu=mu, tau=tau, **otherParams))

    def mu(self, n: Normal, nnn: NnNormal) -> np.ndarray:
        muN, tauN = n.mu, n.tau
        muNnn, tauNnn = nnn.mu, nnn.tau
        tau = tauN + tauNnn
        mu = (muN*tauN + muNnn*tauNnn)/tau
        return(mu)

    def tau(self, n: Normal, nnn: NnNormal) -> np.ndarray:
        tauN = n.tau
        tauNnn = nnn.tau
        tau = tauN + tauNnn

        assertTauIs0 = tf.Assert(tf.reduce_all(tf.logical_not(tf.equal(tau, 0.))), [tau], name='norNnNortauIs0')
        assertTauNotPos = tf.Assert(tf.reduce_all(tf.greater(tau, 0.)), [tau], name='norNnNorCenNotPositive')
        with tf.control_dependencies([assertTauIs0, assertTauNotPos]):
            tau = tau + 0.
        return(tau)
