import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.distribution import Distribution
from decompose.distributions.normal import Normal
from decompose.distributions.nnNormal import NnNormal
from decompose.distributions.exponential import Exponential
from decompose.distributions.product import Product


class NormalExponential(Product):

    def fromUnordered(self, d0: Distribution,
                      d1: Distribution) -> NnNormal:
        if isinstance(d0, Normal) and isinstance(d1, Exponential):
            return(self.product(d0, d1))
        elif isinstance(d1, Normal) and isinstance(d0, Exponential):
            return(self.product(d1, d0))
        else:
            raise ValueError("Expecting Normal and Exponential")

    def product(self, n: Normal, e: Exponential) -> NnNormal:
        mu = self.mu(n, e)
        tau = self.tau(n, e)
        otherParams = self.productParams(n, e)
        return(NnNormal(mu=mu, tau=tau, **otherParams))

    def mu(self, n: Normal, e: Exponential) -> Tensor:
        mu = n.mu - 1. / (n.tau * e.beta)
        return(mu)

    def tau(self, n: Normal, e: Exponential) -> Tensor:
        tau = n.tau * tf.ones_like(e.beta)
        return(tau)
