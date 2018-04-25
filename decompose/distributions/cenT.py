from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

from decompose.distributions.distribution import Distribution
from decompose.distributions.distribution import ParameterInfo
from decompose.distributions.t import T
from decompose.distributions.distribution import parameterProperty
from decompose.distributions.algorithms import Algorithms
from decompose.distributions.cenTAlgorithms import CenTAlgorithms
from decompose.distributions.distribution import Properties


class CenT(T):
    def __init__(self,
                 algorithms: Type[Algorithms] = CenTAlgorithms,
                 Psi: Tensor = None,
                 nu: Tensor = None,
                 tau: Tensor = None,
                 properties: Properties = None) -> None:
        parameters = {"Psi": Psi, "nu": nu, "tau": tau}
        Distribution.__init__(self,
                              algorithms=algorithms,
                              parameters=parameters,
                              properties=properties)

    def parameterInfo(self,
                      shape: Tuple[int, ...] = (1,),
                      latentShape: Tuple[int, ...] = ()) -> ParameterInfo:
        initializers = {
            "Psi": (shape, True),
            "nu": (shape, True),
            "tau": (latentShape + shape, True)
        }  # type: Dict[str, Tensor]
        return(initializers)

    @parameterProperty
    def mu(self) -> Tensor:
        return(tf.zeros_like(self.Psi))
