from typing import Tuple, Callable
import numpy as np
from numpy import ndarray
import tensorflow as tf


class Random(object):
    """Create random data set useful for testing huge input data.

    Arguments:
        M: `Tuple[int, int]`, shape of the data.
        dtype: `type`, type of the training and test data.
    """
    def __init__(self,
                 M: Tuple[int, int] = (1000, 2000),
                 dtype: type = np.float32) -> None:

        self.__M = M
        self.__dtype = tf.as_dtype(dtype)

    @property
    def input_fn(self) -> Callable:
        """Training data as a tensorflow `input_fn` function.

        The whole training matrix will be provided as a minibatch
        and repeated infinitely many times.

        Returns:
            `Callable` that can be passed to the `train` function of
            an `Estimator` function as `input_fn` argument to serve
            the training data.
        """
        def f():
            dataset = tf.data.Dataset.from_tensors(
                 {"data": tf.random_uniform(self.__M, maxval=100,
                                            dtype=self.__dtype)})
            dataset = dataset.repeat()
            return(dataset)
        return(f)

    @property
    def M(self):
        return(self.__M)
