from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.distribution import Distribution
from decompose.distributions.normal import Normal
from decompose.distributions.laplace import Laplace
from decompose.distributions.jumpNormal import JumpNormal
from decompose.distributions.product import Product


class NormalLaplace(Product):

    def fromUnordered(self, d0: Distribution,
                      d1: Distribution) -> Distribution:
        if isinstance(d0, Normal) and isinstance(d1, Laplace):
            return(self.product(d0, d1))
        elif isinstance(d1, Normal) and isinstance(d0, Laplace):
            return(self.product(d1, d0))
        else:
            raise ValueError("Expecting Normal and Exponential")

    def product(self, n: Normal, l: Laplace) -> JumpNormal:
        mu = self.mu(n, l)
        tau = self.tau(n, l)
        nu = self.nu(n, l)
        beta = self.beta(n, l)

        otherParams = self.productParams(n, l)
        return(JumpNormal(mu=mu, tau=tau, nu=nu, beta=beta,
                          **otherParams))

    def mu(self, n: Normal, l: Laplace) -> Tensor:
        mu = n.mu * tf.ones_like(l.mu)
        return(mu)

    def tau(self, n: Normal, l: Laplace) -> Tensor:
        tau = n.tau * tf.ones_like(l.mu)
        return(tau)

    def nu(self, n: Normal, l: Laplace) -> Tensor:
        nu = l.mu * tf.ones_like(n.mu)
        return(nu)

    def beta(self, n: Normal, l: Laplace) -> Tensor:
        beta = l.beta * tf.ones_like(n.mu)
        return(beta)
