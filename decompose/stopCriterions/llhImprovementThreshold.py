import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.stopCriterions.stopCriterion import StopCriterion


class LlhImprovementThreshold(StopCriterion):
    def __init__(self, lhImprovmentThreshold=1.):
        self.__lhImprovmentThreshold = lhImprovmentThreshold

    def init(self, ns: str = "stopCriterion") -> None:
        self.__ns = ns
        with tf.variable_scope(self.__ns):
            llhVar = tf.get_variable("llh",
                                     dtype=tf.float64,
                                     initializer=tf.constant(-np.inf,
                                                             dtype=tf.float64))
            self.llhVar = llhVar
            stopVar = tf.get_variable("stop",
                                      dtype=tf.bool,
                                      initializer=False)
            self.__stopVar = stopVar

    def update(self, model, X: Tensor):
        llh = tf.cast(model.llh(X), tf.float64)
        cond = tf.less(llh - self.llhVar, self.__lhImprovmentThreshold)
        u0 = tf.assign(self.stopVar, cond)
        with tf.control_dependencies([u0]):
            u1 = tf.assign(self.llhVar, llh)
        return([u0, u1])

    @property
    def stopVar(self) -> tf.Variable:
        return(self.__stopVar)
