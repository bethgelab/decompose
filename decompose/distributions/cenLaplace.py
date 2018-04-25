from typing import Tuple, Any, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import ParameterInfo
from decompose.distributions.laplace import Laplace
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.cenLaplaceAlgorithms import CenLaplaceAlgorithms
from decompose.distributions.distribution import Properties


class CenLaplace(Laplace):
    def __init__(self,
                 algorithms: Type[Algorithms] = CenLaplaceAlgorithms,
                 beta: Tensor = None,
                 properties: Properties = None) -> None:
        parameters = {"beta": beta}
        Distribution.__init__(self,
                              algorithms=algorithms,
                              parameters=parameters,
                              properties=properties)

    def parameterInfo(self,
                      shape: Tuple[int, ...] = (1,),
                      latentShape: Tuple[int, ...] = ()) -> ParameterInfo:
        initializers = {
            "beta": (shape, True)
        }  # type: Dict[str, Tensor]
        return(initializers)

    @property
    def mu(self) -> Tensor:
        mu = tf.zeros_like(self.beta)
        return(mu)
