from typing import Tuple, Callable, List
import numpy as np
import string
from numpy import ndarray
import tensorflow as tf


class LowRank(object):
    """Create low rank training and test data.

    Arguments:
        rank: `int`, rank of the data.
        M_train: `Tuple[int, ...]`, shape of the training data.
        M_test: `Tuple[int, ...]`, shape of the test data.
        dtype: `type`, type of the training and test data.

    Raises:
        ValueError:
            if `M_train` and `M_test` differ in any but the first element.
    """
    def __init__(self,
                 rank: int = 3,
                 M_train: Tuple[int, ...] = (2000, 1000),
                 M_test: Tuple[int, ...] = (500, 1000),
                 dtype: type = np.float32) -> None:

        if M_train[1:] != M_test[1:]:
            raise ValueError

        U0_train = np.random.normal(size=(rank, M_train[0]))
        U0_test = np.random.normal(size=(rank, M_test[0]))
        UsList = []  # type: List[np.ndarray]
        for M in M_train[1:]:
            UsList.append(np.random.normal(size=(rank, M)))
        Us = tuple(UsList)
        signal_train = self.tensorReconstruction((U0_train,) + Us)
        noise_train = np.random.normal(size=M_train, scale=0.1)
        data_train = signal_train + noise_train
        signal_test = self.tensorReconstruction((U0_test,) + Us)
        noise_test = np.random.normal(size=M_test, scale=0.1)
        data_test = signal_test + noise_test
        self.__Us = Us
        self.__U0_train = U0_train
        self.__data_train = data_train.astype(dtype)
        self.__U0_test = U0_test
        self.__data_test = data_test.astype(dtype)
        self.__varTraining = np.var(data_train)
        self.__varTest = np.var(data_test)

    def tensorReconstruction(self, U: Tuple[ndarray, ndarray]) -> ndarray:
        """Reconstructs the data using the estimates provided"""
        F = len(U)
        axisIds = string.ascii_lowercase[:F]
        subscripts = f'k{",k".join(axisIds)}->{axisIds}'
        r = np.einsum(subscripts, *U)
        return(r)

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
        r = self.tensorReconstruction(U) - self.training
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
        r = self.tensorReconstruction(U) - self.test
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
