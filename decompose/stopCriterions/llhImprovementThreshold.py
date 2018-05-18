import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from decompose.stopCriterions.stopCriterion import StopCriterion


class LlhImprovementThreshold(StopCriterion):
    def __init__(self, lhImprovementThreshold: float = 1.) -> None:
        self.__lhImprovementThreshold = np.float64(lhImprovementThreshold)

    def init(self, ns: str = "stopCriterion") -> None:
        self.__ns = ns
        negInf = tf.constant(-np.inf, dtype=tf.float64)
        with tf.variable_scope(self.__ns):
            llhVar = tf.get_variable("llh",
                                     dtype=tf.float64,
                                     initializer=negInf)
            self.llhVar = llhVar
            stopVar = tf.get_variable("stop",
                                      dtype=tf.bool,
                                      initializer=False)
            self.__stopVar = stopVar

    def update(self, model, X: Tensor):
        llh = tf.cast(model.llh(X), tf.float64)
        llhOld = self.llhVar
        cond = tf.greater(self.llhImprovementThreshold, llh - llhOld)
        u0 = tf.assign(self.stopVar, cond)
        with tf.control_dependencies([u0]):
            u1 = tf.assign(self.llhVar, llh)
        return([u0, u1])

    @property
    def stopVar(self) -> tf.Variable:
        return(self.__stopVar)

    @property
    def llhImprovementThreshold(self) -> float:
        return(self.__lhImprovementThreshold)
