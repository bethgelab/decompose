from typing import Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.tAlgorithms import TAlgorithms


class CenTAlgorithms(TAlgorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        Psi, nu = parameters["Psi"], parameters["nu"]
        mu = tf.zeros_like(Psi)
        parameters = {"nu": nu, "mu": mu, "Psi": Psi}
        samples = TAlgorithms.sample(parameters=parameters,
                                     nSamples=nSamples)
        return(samples)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        Psi = parameters["Psi"]
        mode = tf.zeros_like(Psi)
        return(mode)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        Psi, nu = parameters["Psi"], parameters["nu"]
        mu = tf.zeros_like(Psi)
        parameters = {"nu": nu, "mu": mu, "Psi": Psi}
        pdf = TAlgorithms.pdf(parameters=parameters, data=data)
        return(pdf)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> Tensor:
        Psi, nu = parameters["Psi"], parameters["nu"]
        mu = tf.zeros_like(Psi)
        parameters = {"nu": nu, "mu": mu, "Psi": Psi}
        llh = TAlgorithms.llh(parameters=parameters, data=data)
        return(llh)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        Psi, nu = parameters["Psi"], parameters["nu"]

        p = 1.

        n = data.get_shape()[0].value
        Y = data
        for i in range(5):
            delta = Y**2/Psi
            w = (nu + p)/(nu + delta)

            PsiNew = tf.reduce_mean(w*Y**2, axis=0)
            cond = tf.logical_and(tf.is_finite(PsiNew),
                                  tf.greater(PsiNew, 1e-6))
            Psi = tf.where(cond, PsiNew, Psi*tf.ones_like(PsiNew))

            delta = Y**2/Psi
            nuNew = cls.nuStep(nu, n, delta)
            cond = tf.logical_and(tf.is_finite(nuNew),
                                  tf.greater(nuNew, 0.))
            nu = tf.where(cond, nuNew, nu*tf.ones_like(nuNew))

        tau = w
        updatedParameters = {"nu": nu, "Psi": Psi, "tau": tau}
        return(updatedParameters)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        Psi, nu = parameters["Psi"], parameters["nu"]
        p = 1.
        Y = data
        delta = Y**2/Psi
        w = (nu + p)/(nu + delta)
        return({"tau": w})
