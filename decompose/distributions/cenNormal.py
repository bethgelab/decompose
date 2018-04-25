from typing import Tuple, Any, Dict, Type
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import ParameterInfo
from decompose.distributions.normal import Normal
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.cenNormalAlgorithms import CenNormalAlgorithms
from decompose.distributions.distribution import Properties


class CenNormal(Normal):
    def __init__(self,
                 algorithms: Type[Algorithms] = CenNormalAlgorithms,
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
