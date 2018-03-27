import numpy as np
import tensorflow as tf
from tensorflow import Tensor


class LlhStall(object):
    def __init__(self, nStalledIterationsThreshold: int = 100,
                 ns: str = "stopCriterion") -> None:
        self.__nStalledIterationsThrehold = nStalledIterationsThreshold
        self.__ns = ns

    def init(self):
        llhsInit = -np.inf*tf.ones(self.__nStalledIterationsThrehold,
                                   dtype=tf.float64)
        with tf.variable_scope(self.__ns):
            llhsVar = tf.get_variable("llhs",
                                      dtype=tf.float64,
                                      initializer=llhsInit)
            self.llhsVar = llhsVar
            stopVar = tf.get_variable("stop",
                                      dtype=tf.bool,
                                      initializer=False)
            self.__stopVar = stopVar

    def update(self, model, X: Tensor):
        llh = tf.cast(model.llh(X), tf.float64)
        llhsVar = self.llhsVar
        llhsUpdated = tf.concat((llh[None], llhsVar[1:]), axis=0)
        llhsUpdated = tf.manip.roll(llhsUpdated, shift=1, axis=0)
        cond = tf.reduce_all(tf.greater(llhsUpdated[0], llhsUpdated[1:]))

        u0 = tf.assign(self.stopVar, cond)
        with tf.control_dependencies([u0]):
            u1 = tf.assign(self.llhsVar, llhsUpdated)
        return([u0, u1])

    @property
    def stopVar(self) -> tf.Variable:
        return(self.__stopVar)
