from typing import Tuple, Dict
import tensorflow as tf
from tensorflow import Tensor


class Algorithms(object):

    def __init__(self):
        return

    @classmethod
    def sample(cls, parameters: Dict[str, Tensor], nSamples: Tensor) -> Tensor:
        # here the sample axis is postpended unlike tf where it is postpend
        ...

    @classmethod
    def mode(cls, parameters: Dict[str, Tensor]) -> Tensor:
        ...

    @classmethod
    def pdf(cls, parameters: Dict[str, Tensor], data: Tensor) -> Tensor:
        ...

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: Tensor) -> Dict[str, Tensor]:
        ...

    @classmethod
    def llh(cls, parameters: Dict[str, Tensor], data: tf.Tensor) -> float:
        ...

    @classmethod
    def fitLatents(cls, parameters: Dict[str, Tensor],
                   data: Tensor) -> Dict[str, Tensor]:
        ...
