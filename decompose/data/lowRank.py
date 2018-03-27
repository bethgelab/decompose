from typing import Tuple, Callable
import numpy as np
from numpy import ndarray
import tensorflow as tf


class LowRank(object):
    """Create low rank training and test data.

    Arguments:
        rank: `int`, rank of the data.
        M_train: `Tuple[int, int]`, shape of the training data.
        M_test: `Tuple[int, int]`, shape of the test data.
        dtype: `type`, type of the training and test data.

    Raises:
        ValueError:
            if `M_train` and `M_test` differ in the second element.
    """
    def __init__(self,
                 rank: int = 3,
                 M_train: Tuple[int, int] = (2000, 1000),
                 M_test: Tuple[int, int] = (500, 1000),
                 dtype: type = np.float32) -> None:

        if M_train[1] != M_test[1]:
            raise ValueError

        U0_train = np.random.normal(size=(rank, M_train[0]))
        U0_test = np.random.normal(size=(rank, M_test[0]))
        U1 = np.random.normal(size=(rank, M_train[1]))
        signal_train = np.dot(U0_train.T, U1)
        noise_train = np.random.normal(size=M_train, scale=0.1)
        data_train = signal_train + noise_train
        signal_test = np.dot(U0_test.T, U1)
        noise_test = np.random.normal(size=M_test, scale=0.1)
        data_test = signal_test + noise_test
        self.__U1 = U1
        self.__U0_train = U0_train
        self.__data_train = data_train.astype(dtype)
        self.__U0_test = U0_test
        self.__data_test = data_test.astype(dtype)
        self.__varTraining = np.var(data_train)
        self.__varTest = np.var(data_test)

    @property
    def training(self) -> ndarray:
        """Training data as a numpy array.

        Returns:
            `ndarray` of shape `M_train`."""
        return(self.__data_train)

    @property
    def test(self) -> ndarray:
        """Test data as a numpy array.

        Returns:
            `ndarray` of shape `M_test`."""
        return(self.__data_test)

    @property
    def training_input_fn(self) -> Callable:
        """Training data as a tensorflow `input_fn` function.

        The whole training matrix will be provided as a minibatch
        and repeated infinitely many times.

        Returns:
            `Callable` that can be passed to the `train` function of
            an `Estimator` function as `input_fn` argument to serve
            the training data.
        """
        npData = self.__data_train
        x = {"train": npData}
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x, y=None, batch_size=npData.shape[0],
            shuffle=False, num_epochs=None)
        return(input_fn)

    @property
    def test_input_fn(self) -> Callable:
        """Test data as a tensorflow `input_fn` function.

        The whole test matrix will be provided as a minibatch
        and repeated infinitely many times.

        Returns:
            `Callable` that can be passed to the `train` function of
            an `Estimator` function as `input_fn` argument to serve
            the training data.
        """
        npData = self.__data_test
        x = {"test": npData}
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x, y=None, batch_size=npData.shape[0],
            shuffle=False, num_epochs=None)
        return(input_fn)

    @property
    def __var_training(self) -> float:
        """Variance of the training data.

            Returns:
            `float` variance of the training data."""
        return(self.__varTraining)

    @property
    def __var_test(self) -> float:
        """Variance of the test data.

            Returns:
            `float` variance of the test data."""
        return(self.__varTest)

    def residuals_training(self, U: Tuple[ndarray, ndarray]) -> float:
        """Residuals of the training data for a given filter banks `U` .

        The residuals between the training data and its approximation
        using the filter banks `U` .

        Arguments:
            U: `Tuple[ndarray, ndarray]` filter banks.

        Returns:
            `ndarray` of shape `M_train` with the difference between the
            reconstruction and the data.
        """
        r = np.dot(U[0].T, U[1]) - self.training
        return(r)

    def var_expl_training(self, U: Tuple[ndarray, ndarray]) -> float:
        """Variance explained of the training data given filter banks `U` .

        Arguments:
            U: `Tuple[ndarray, ndarray]` tupel of filter banks.

        Returns:
            `float` fraction of the training data variance that
            is exaplained by the filter banks `U` .
        """
        varExpl = 1. - (np.var(self.residuals_training(U))
                        / self.__var_training)
        return(varExpl)

    def residuals_test(self, U: Tuple[ndarray, ndarray]) -> float:
        """Residuals of the test data for a given filter banks `U` .

        The residuals between the training data and its approximation
        using the provided filter banks `U` .

        Arguments:
            U: `Tuple[ndarray, ndarray]` filter banks.

        Returns:
            `ndarray` of shape `M_test` with the difference between the
            reconstruction and the data.
        """
        r = np.dot(U[0].T, U[1]) - self.test
        return(r)

    def var_expl_test(self, U: Tuple[ndarray, ndarray]) -> float:
        """Variance explained of the test data using provided filter banks.

        Arguments:
            U: `Tuple[ndarray, ndarray]` filter banks.

        Returns:
            `float` fraction of the training data variance that
            is exaplained by the filter banks `U` .
        """
        varExpl = 1. - (np.var(self.residuals_test(U))
                        / self.__var_test)
        return(varExpl)
