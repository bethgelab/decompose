from typing import Tuple, Any, Dict, Type
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import ParameterInfo
from decompose.distributions.nnNormal import NnNormal
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.cenNnNormalAlgorithms import CenNnNormalAlgorithms
from decompose.distributions.distribution import Properties


class CenNnNormal(NnNormal):
    def __init__(self,
                 algorithms: Type[Algorithms] = CenNnNormalAlgorithms,
                 tau: Tensor = None,
                 properties: Properties = None) -> None:
        parameters = {"tau": tau}
        Distribution.__init__(self,
                              algorithms=algorithms,
                              parameters=parameters,
                              properties=properties)

    def parameterInfo(self,
                      shape: Tuple[int, ...] = (1,),
                      latentShape: Tuple[int, ...] = ()) -> ParameterInfo:
        initializers = {
            "tau": (shape, True)
        }  # type: Dict[str, Tensor]
        return(initializers)

    @property
    def mu(self) -> Tensor:
        mu = tf.zeros_like(self.tau)
        return(mu)
