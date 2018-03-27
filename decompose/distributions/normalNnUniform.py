from typing import Tuple
import tensorflow as tf
import numpy as np
from tensorflow import Tensor

from decompose.distributions.distribution import Distribution
from decompose.distributions.normal import Normal
from decompose.distributions.nnNormal import NnNormal
from decompose.distributions.nnUniform import NnUniform
from decompose.distributions.product import Product


class NormalNnUniform(Product):

    def fromUnordered(self, d0: Distribution,
                      d1: Distribution) -> NnNormal:
        if isinstance(d0, Normal) and isinstance(d1, NnUniform):
            return(self.product(d0, d1))
        elif isinstance(d1, Normal) and isinstance(d0, NnUniform):
            return(self.product(d1, d0))
        else:
            raise ValueError("Expecting Normal and NnUniform")

    def product(self, n: Normal, u: NnUniform) -> NnNormal:
        mu = self.mu(n, u)
        tau = self.tau(n, u)
        otherParams = self.productParams(n, u)
        return(NnNormal(mu=mu, tau=tau, **otherParams))

    def mu(self, n, u) -> Tensor:
        ones = tf.ones(u.shape, dtype=n.mu.dtype)
        return(n.mu*ones)

    def tau(self, n, u) -> Tensor:
        ones = tf.ones(u.shape, dtype=n.mu.dtype)
        return(n.tau*ones)
