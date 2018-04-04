from typing import Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms
from decompose.distributions.cenTAlgorithms import CenTAlgorithms


class CenNnTAlgorithms(Algorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        Psi, nu = parameters["Psi"], parameters["nu"]
        mu = tf.zeros_like(Psi)
        t = tf.distributions.StudentT(df=nu, loc=mu, scale=tf.sqrt(Psi))
        r = t.sample(sample_shape=(nSamples,))
        return(r)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mode = tf.zeros_like(parameters["Psi"])
        return(mode)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        Psi, nu = parameters["Psi"], parameters["nu"]
        mu = tf.zeros_like(Psi)
        t = tf.distributions.StudentT(df=nu, loc=mu, scale=tf.sqrt(Psi))
        pdf = 2*t.prob(value=data)
        pdf = tf.where(tf.less(data, 0.), tf.zeros_like(pdf), pdf)
        return(pdf)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> Tensor:
        Psi, nu = parameters["Psi"], parameters["nu"]
        mu = tf.zeros_like(Psi)
        t = tf.distributions.StudentT(df=nu, loc=mu, scale=tf.sqrt(Psi))
        llh = t.log_prob(value=data)
        llh = tf.where(tf.less(data, 0.), -np.inf*tf.ones_like(llh), llh)
        return(llh)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        return(CenTAlgorithms.fit(parameters, data))

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        return(CenTAlgorithms.fitLatents(parameters, data))
