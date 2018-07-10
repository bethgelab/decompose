from typing import Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms


class LomaxAlgorithms(Algorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        alpha, beta = parameters["alpha"], parameters["beta"]
        gamma = tf.distributions.Gamma(concentration=alpha, rate=beta)
        tau = gamma.sample(sample_shape=(nSamples,))
        exp = tf.distributions.Exponential(rate=tau)
        s = exp.sample()
        return(s)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mode = tf.zeros_like(parameters["alpha"])
        return(mode)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        return(tf.exp(cls.llh(parameters=parameters, data=data)))

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> Tensor:
        alpha, beta = parameters["alpha"], parameters["beta"]
        llh = tf.log(alpha) - tf.log(beta) - (alpha+1)*tf.log(1.+data/beta)
        llh = tf.where(tf.less(data, 0.), -np.inf*tf.ones_like(llh), llh)
        return(llh)

    @classmethod
    def f(cls, y, beta, tn):
        tal = tn/tf.reduce_sum(tf.log((beta+y)/beta), axis=-2, keepdims=True)
        tfu = tn*(tf.log(tal)-tf.log(beta) - (1.+1./tal))
        return(tfu)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        alphaOld, betaOld = parameters["alpha"], parameters["beta"]

        y = data
        tn = data.get_shape()[0].value
        alpha = alphaOld
        beta = betaOld

        for i in range(1):
            swidths = tf.constant(np.array([0., 1e-8, 1e-7, 1e-6, 1e-4, 1e-3,
                                            1e-2, 1e-1, 1e0, 1e1, 1e2,
                                            1e3]), dtype=alpha.dtype)
            grads = tf.gradients(cls.f(data, beta, tn), [beta])[0]
            normedDirection = tf.sign(grads)
            testBetas = beta+swidths[..., None, None]*normedDirection
            testUtilities = cls.f(data, testBetas, tn)
            argmax = tf.cast(tf.argmax(testUtilities, axis=0)[0],
                             dtype=tf.int32)
            maxWidths = tf.gather(swidths, argmax)
            beta = beta+maxWidths*normedDirection

            beta = tf.where(tf.greater(beta, 1e-9), beta,
                            1e-9*tf.ones_like(beta))
            beta = tf.where(tf.less(beta, 1e9), beta,
                            1e9*tf.ones_like(beta))

        alpha = tn/tf.reduce_sum(tf.log(beta+y) - tf.log(beta), axis=-2)
        alpha = tf.where(tf.greater(alpha, 1e-9), alpha,
                         1e-9*tf.ones_like(alpha))
        alpha = tf.where(tf.less(alpha, 1e9), alpha,
                         1e9*tf.ones_like(alpha))

        llh = tf.reduce_mean(cls.llh(parameters={"alpha": alpha, "beta": beta},
                                     data=data), axis=0)
        llhOld = tf.reduce_mean(cls.llh(parameters={"alpha": alphaOld, "beta": betaOld},
                                        data=data), axis=0)
        alpha = tf.where(llh > llhOld,
                         alpha,
                         alphaOld)
        beta = tf.where(llh > llhOld,
                        beta,
                        betaOld)
        w = (alpha + 1)/(beta + y)
        return({"alpha": alpha,
                "beta": beta,
                "tau": w})

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        alpha, beta = parameters["alpha"], parameters["beta"]
        tau = (alpha + 1)/(beta + data)
        return({"tau": tau})
