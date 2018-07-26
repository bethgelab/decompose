from typing import Tuple, Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.algorithms import Algorithms
from decompose.distributions.lomaxAlgorithms import LomaxAlgorithms
from decompose.distributions.exponentialAlgorithms import ExponentialAlgorithms
from decompose.distributions.cenNnElasticNetAlgorithms import CenNnElasticNetAlgorithms


class CenNnFullyElasticNetAlgorithms(Algorithms):

    @classmethod
    def getParameters(cls, parameters: Dict[str, Tensor]) -> Tuple[Tensor,
                                                                   Dict[str, Tensor],
                                                                   Dict[str, Tensor],
                                                                   Dict[str, Tensor]]:
        b = parameters['b']
        cenNnElasticnetParameters = {"mu": parameters["mu"],
                                     "tau": parameters["tau"]}
        exponentialParameters = {"beta": parameters["betaExponential"]}
        lomaxParameters = {"mu": tf.zeros_like(parameters["mu"]),
                           "alpha": parameters["alpha"],
                           "beta": parameters["beta"],
                           "tau": parameters["tauLomax"]}
        return(b, cenNnElasticnetParameters, exponentialParameters,
               lomaxParameters)

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        params = cls.getParameters(parameters=parameters)
        b, cenNnElasticnetParams, exponentialParams, lomaxParams = params

        rLomax = LomaxAlgorithms.sample(lomaxParams, nSamples)
        rExponential = ExponentialAlgorithms.sample(exponentialParams, nSamples)
        rElasticNet = CenNnElasticNetAlgorithms.sample(cenNnElasticnetParams, nSamples)
        b = b * tf.ones_like(rLomax)
        r = tf.where(tf.equal(b, 0.), rElasticNet,
                     tf.where(tf.equal(b, 1.), rExponential, rLomax))

        return(r)

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        mu = tf.zeros_like(parameters["mu"])
        return(mu)

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        params = cls.getParameters(parameters=parameters)
        b, cenNnElasticnetParams, exponentialParams, lomaxParams = params

        pdfLomax = LomaxAlgorithms.pdf(lomaxParams, data)
        pdfExponential = ExponentialAlgorithms.pdf(exponentialParams, data)
        pdfElasticNet = CenNnElasticNetAlgorithms.pdf(cenNnElasticnetParams,
                                                      data)
        b = b[None]*tf.ones_like(pdfElasticNet)
        pdf = tf.where(tf.equal(b, 0.), pdfElasticNet,
                       tf.where(tf.equal(b, 1.), pdfExponential, pdfLomax))
        return(pdf)

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        params = cls.getParameters(parameters=parameters)
        b, cenNnElasticnetParams, exponentialParams, lomaxParams = params

        cenNnElasticnetParams = CenNnElasticNetAlgorithms.fit(
            cenNnElasticnetParams, data)
        exponentialParams = ExponentialAlgorithms.fit(exponentialParams, data)
        lomaxParams = LomaxAlgorithms.fit(lomaxParams, data)

        cenNnElasticnetLlh = CenNnElasticNetAlgorithms.llh(
            cenNnElasticnetParams, data)
        cenNnElasticnetLlh = tf.reduce_mean(cenNnElasticnetLlh, axis=0)
        exponentialLlh = ExponentialAlgorithms.llh(exponentialParams, data)
        exponentialLlh = tf.reduce_mean(exponentialLlh, axis=0)
        lomaxLlh = LomaxAlgorithms.llh(lomaxParams, data)
        lomaxLlh = tf.reduce_mean(lomaxLlh, axis=0)

        condElasticNet = tf.logical_and(cenNnElasticnetLlh > lomaxLlh,
                                        cenNnElasticnetLlh > exponentialLlh)
        condExponential = exponentialLlh > lomaxLlh

        b = tf.where(condElasticNet,
                     tf.zeros_like(cenNnElasticnetLlh),
                     tf.where(condExponential,
                              tf.ones_like(exponentialLlh),
                              2.*tf.ones_like(lomaxLlh)))

        updatedParameters = {"b": b,
                             "mu": cenNnElasticnetParams["mu"],
                             "tau": cenNnElasticnetParams["tau"],
                             "betaExponential": exponentialParams["beta"],
                             "alpha": lomaxParams["alpha"],
                             "beta": lomaxParams["beta"],
                             "tauLomax": lomaxParams["tau"]}
        return(updatedParameters)

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> float:
        params = cls.getParameters(parameters=parameters)
        b, cenNnElasticnetParams, exponentialParams, lomaxParams = params

        llhLomax = LomaxAlgorithms.llh(lomaxParams, data)
        llhExponential = ExponentialAlgorithms.llh(exponentialParams, data)
        llhElasticNet = CenNnElasticNetAlgorithms.llh(cenNnElasticnetParams,
                                                      data)
        b = b[None]*tf.ones_like(llhElasticNet)
        llh = tf.where(tf.equal(b, 0.), llhElasticNet,
                       tf.where(tf.equal(b, 1.), llhExponential, llhLomax))
        return(llh)

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        params = cls.getParameters(parameters=parameters)
        b, cenNnElasticnetParams, exponentialParams, lomaxParams = params

        lomaxParamsUp = LomaxAlgorithms.fitLatents(lomaxParams, data)
        lomaxParams["tau"] = lomaxParamsUp["tau"]

        updatedParams = {"tauLomax": lomaxParams["tau"]}

        return(updatedParams)
