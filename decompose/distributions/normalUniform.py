from typing import Tuple
import tensorflow as tf
import numpy as np
from tensorflow import Tensor

from decompose.distributions.distribution import Distribution
from decompose.distributions.normal import Normal
from decompose.distributions.uniform import Uniform
from decompose.distributions.product import Product


class NormalUniform(Product):

    def fromUnordered(self, d0: Distribution,
                      d1: Distribution) -> Normal:
        if isinstance(d0, Normal) and isinstance(d1, Uniform):
            return(self.product(d0, d1))
        elif isinstance(d1, Normal) and isinstance(d0, Uniform):
            return(self.product(d1, d0))
        else:
            raise ValueError("Expecting Normal and Uniform")

    def product(self, n: Normal, u: Uniform) -> Normal:
        mu = self.mu(n, u)
        tau = self.tau(n, u)
        otherParams = self.productParams(n, u)
        return(Normal(mu=mu, tau=tau, **otherParams))

    def mu(self, n, u) -> Tensor:
        ones = tf.ones(u.shape, dtype=n.mu.dtype)
        return(n.mu*ones)

    def tau(self, n, u) -> Tensor:
        ones = tf.ones(u.shape, dtype=n.mu.dtype)
        return(n.tau*ones)
