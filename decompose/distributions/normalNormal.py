from typing import Tuple
import numpy as np
from tensorflow import Tensor

from decompose.distributions.distribution import Distribution
from decompose.distributions.normal import Normal
from decompose.distributions.product import Product


class NormalNormal(Product):

    def fromUnordered(self, d0: Distribution,
                      d1: Distribution) -> Normal:

        if isinstance(d0, Normal) and isinstance(d1, Normal):
            return(self.product(d0, d1))
        else:
            raise ValueError("Expecting Normal and Normal")

    def product(self, n0: Normal, n1: Normal) -> Normal:
        mu = self.mu(n0, n1)
        tau = self.tau(n0, n1)
        otherParams = self.productParams(n0, n1)
        pd = Normal(mu=mu, tau=tau, **otherParams)
        return(pd)

    def mu(self, n0, n1) -> Tensor:
        mu0, tau0 = n0.mu, n0.tau
        mu1, tau1 = n1.mu, n1.tau
        tau = self.tau(n0, n1)
        mu = (mu0*tau0 + mu1*tau1)/tau
        return(mu)

    def tau(self, n0, n1) -> Tensor:
        tau = n0.tau + n1.tau
        return(tau)
