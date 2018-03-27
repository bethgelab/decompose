from typing import Tuple, Dict
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.nnNormalAlgorithms import NnNormalAlgorithms


class CenNnNormalAlgorithms(NnNormalAlgorithms):

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        tau = parameters["tau"]
        mu = tf.zeros_like(tau)
        samples = NnNormalAlgorithms.sample(parameters={"mu": mu, "tau": tau},
                                            nSamples=nSamples)
        return(samples)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        tau = parameters["tau"]
        mode = tf.zeros_like(tau)
        return(mode)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        tau = parameters["tau"]
        mu = tf.zeros_like(tau)
        pdf = NnNormalAlgorithms.pdf(parameters={"mu": mu, "tau": tau},
                                     data=data)
        return(pdf)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        var = tf.reduce_mean(data**2, axis=-1)
        tau = 1./var
        updatedParameters = {"tau": tau}
        return(updatedParameters)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> float:
        tau = parameters["tau"]
        mu = tf.zeros_like(tau)

        llh = NnNormalAlgorithms.llh(parameters={"mu": mu, "tau": tau},
                                     data=data)
        return(llh)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        return({})
