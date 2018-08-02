from typing import Tuple
import numpy as np
import tensorflow as tf

from decompose.distributions.distribution import Distribution
from decompose.distributions.normal import Normal
from decompose.distributions.nnNormal import NnNormal
from decompose.distributions.exponential import Exponential
from decompose.distributions.cenNnFullyElasticNetCond import CenNnFullyElasticNetCond
from decompose.distributions.product import Product


class NormalCenNnFullyElasticNetCond(Product):

    def fromUnordered(self, d0: Distribution,
                      d1: Distribution) -> Normal:

        if isinstance(d0, Normal) and isinstance(d1, CenNnFullyElasticNetCond):
            return(self.product(d0, d1))
        else:
            raise ValueError("Expecting Normal and CenNnFullyElasticNet")

    def product(self, n0: Normal, n1: CenNnFullyElasticNetCond) -> NnNormal:
        otherParams = self.productParams(n0, n1)

        lomax = Exponential(beta=n1.beta, **otherParams)
        normalLomax = n0*lomax

        exponential = Exponential(beta=n1.betaExponential, **otherParams)
        normalExponential = n0*exponential

        nnNormal = NnNormal(mu=n1.mu, tau=n1.tau, **otherParams)
        normalNnNormal = n0*nnNormal

        b = n1.b
        mu = tf.where(tf.equal(b, 0.),
                      normalNnNormal.mu,
                      tf.where(tf.equal(b, 1.),
                               normalExponential.mu,
                               normalLomax.mu))
        tau = tf.where(tf.equal(b, 0.),
                       normalNnNormal.tau,
                       tf.where(tf.equal(b, 1.),
                                normalExponential.tau,
                                normalLomax.tau))

        pd = NnNormal(mu=mu, tau=tau, **otherParams)
        return(pd)
