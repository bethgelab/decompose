from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import Tensor


class StopCriterion(ABC):

    @abstractmethod
    def init(self, ns: str) -> None:
        ...


class NoStop(StopCriterion):
    def __init__(self):
        pass

    def init(self, ns="stopCriterion"):
        self.__ns = ns
        with tf.variable_scope(self.__ns):
            stopVar = tf.get_variable("stop",
                                      dtype=tf.bool,
                                      initializer=False)
            self.__stopVar = stopVar

    def update(self, model, X: Tensor):
        return([])

    @property
    def stopVar(self) -> tf.Variable:
        return(self.__stopVar)


class StopHook(tf.train.SessionRunHook):
    def after_run(self, run_context, run_values):
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            stopVar = tf.get_variable("stopCriterion/stop",
                                      dtype=tf.bool,
                                      shape=())
            condition = run_context.session.run(stopVar)
        if condition:
            run_context.request_stop()
