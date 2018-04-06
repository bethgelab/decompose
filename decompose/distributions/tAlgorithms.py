from typing import Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms


class TAlgorithms(Algorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        mu, Psi, nu = parameters["mu"], parameters["Psi"], parameters["nu"]
        t = tf.distributions.StudentT(df=nu, loc=mu, scale=tf.sqrt(Psi))
        r = t.sample(sample_shape=(nSamples,))
        return(r)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mu = parameters["mu"]
        return(mu)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        mu, Psi, nu = parameters["mu"], parameters["Psi"], parameters["nu"]
        t = tf.distributions.StudentT(df=nu, loc=mu, scale=tf.sqrt(Psi))
        pdf = t.prob(value=data)
        return(pdf)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> Tensor:
        mu, Psi, nu = parameters["mu"], parameters["Psi"], parameters["nu"]
        t = tf.distributions.StudentT(df=nu, loc=mu, scale=tf.sqrt(Psi))
        llh = t.log_prob(value=data)
        return(llh)

    @classmethod
    def nuStep(cls, nu, n, delta, p=1.):
        three = tf.constant(3., dtype=nu.dtype)
        for i in range(2):
            w = (nu+p)/(nu+delta)
            fp = (-tf.digamma(nu/2) + tf.log(nu/2)
                  + 1./n*tf.reduce_sum(tf.log((nu+p)/(nu+delta)) - w,
                                       axis=0)
                  + 1
                  + tf.digamma((p+nu)/2) - tf.log((p+nu)/2))
            fpp = (tf.polygamma(three, nu/2)/2. + 1./nu
                   + tf.polygamma(three, (p+nu)/2)/2. - 1./(nu+p)
                   + 1./n*tf.reduce_sum((delta-p)/(nu+delta)**2*(w-1),
                                        axis=0))
            nu = nu + fp/fpp
        return(nu)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        mu, Psi, nu = parameters["mu"], parameters["Psi"], parameters["nu"]

        p = 1.

        n = data.get_shape()[0].value
        Y = data
        for i in range(5):
            delta = (Y - mu)**2/Psi
            w = (nu + p)/(nu + delta)

            mu = tf.reduce_mean(w*Y, axis=0)
            PsiNew = tf.reduce_mean(w*(Y-mu)**2, axis=0)
            cond = tf.logical_and(tf.is_finite(PsiNew),
                                  tf.greater(PsiNew, 1e-6))
            Psi = tf.where(cond, PsiNew, Psi*tf.ones_like(PsiNew))

            delta = (Y - mu)**2/Psi
            nuNew = cls.nuStep(nu, n, delta)
            cond = tf.logical_and(tf.is_finite(nuNew),
                                  tf.greater(nuNew, 0.))
            nu = tf.where(cond, nuNew, nu*tf.ones_like(nuNew))

        tau = w
        updatedParameters = {"mu": mu, "nu": nu, "Psi": Psi, "tau": tau}
        return(updatedParameters)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        mu, Psi, nu = parameters["mu"], parameters["Psi"], parameters["nu"]
        p = 1.
        Y = data
        delta = (Y - mu)**2/Psi
        w = (nu + p)/(nu + delta)
        return({"tau": w})
