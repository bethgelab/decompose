from typing import Tuple, Dict
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.distributions.cenNormalRankOneAlgorithms import CenNormalRankOneAlgorithms


class CenNormalRankOneHeuristicAlgorithms(CenNormalRankOneAlgorithms):

    @classmethod
    def fit(cls, parameters: Dict[str, Tensor],
            data: tf.Tensor) -> Dict[str, Tensor]:
        """Non-optimal update Using an heuristic."""

        var0 = tf.sqrt(tf.reduce_mean(data**2, axis=1))
        var1 = tf.sqrt(tf.reduce_mean(data**2, axis=0))
        c = tf.reduce_mean(data**2)/tf.reduce_mean(tf.matmul(var0[..., None],
                                                             var1[None, ...]))
        tau0 = 1./(var0*tf.sqrt(c))
        tau1 = 1./(var1*tf.sqrt(c))

        updatedParameters = {"tau0": tau0, "tau1": tau1}
        return(updatedParameters)
