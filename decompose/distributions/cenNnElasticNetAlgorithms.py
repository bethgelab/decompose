from typing import Dict
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.nnNormalAlgorithms import NnNormalAlgorithms


class CenNnElasticNetAlgorithms(NnNormalAlgorithms):

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mode = tf.zeros_like(parameters["mu"])
        return(mode)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        # get non-negative normal fit
        nnNormalParameters = NnNormalAlgorithms.fit(parameters=parameters,
                                                    data=data)
        mu, tau = nnNormalParameters["mu"], nnNormalParameters["tau"]

        # project mus on the non-positive interval and update taus accordingly
        projectMuToZero = tf.greater(mu, 0)
        mu = tf.where(projectMuToZero, tf.zeros_like(mu), mu)
        tauGivenMuIsZero = 1./tf.reduce_mean(data**2, axis=0)
        tau = tf.where(projectMuToZero, tauGivenMuIsZero, tau)
        updatedParameters = {"mu": mu, "tau": tau}

        return(updatedParameters)
