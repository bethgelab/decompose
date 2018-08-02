from typing import Tuple
from abc import ABCMeta, abstractmethod
import string
import numpy as np
import tensorflow as tf
from tensorflow import Tensor


class CV(metaclass=ABCMeta):
    @abstractmethod
    def isLowrank(self) -> bool:
        ...

    @abstractmethod
    def lowrankMask(self, X: Tensor) -> Tuple[Tensor, ...]:
        ...

    @abstractmethod
    def mask(self, X: Tensor) -> Tensor:
        ...


class Block(CV):

    def __init__(self, nFolds: Tuple[int, ...], foldNumber: int) -> None:
        self.__nFolds = nFolds
        self.__foldNumber = foldNumber

    @property
    def nFolds(self) -> Tuple[int, ...]:
        return(self.__nFolds)

    @property
    def foldNumber(self) -> int:
        return(self.__foldNumber)

    def isLowrank(self) -> bool:
        return(True)

    def lowrankMask(self, X: Tensor):
        nFolds = np.array(self.nFolds)
        foldNumber = self.foldNumber

        M = np.array(X.get_shape().as_list())
        F = len(M)
        nValues = M//nFolds

        folds = np.zeros(np.product(M)).flatten()
        folds[foldNumber] = 1.
        folds = folds.reshape(M)
        foldNumbers = np.array(np.where(folds == 1.)).flatten()

        U = []
        for f in range(F):
            Uf = self.testMask(M[f], foldNumbers[f], nFolds[f], nValues[f])
            U.append(tf.constant(Uf))
        return(U)

    def mask(self, X: Tensor):
        U = self.lowrankMask(X)
        F = len(U)
        axis = string.ascii_lowercase[:F]
        subscripts = ','.join(axis) + "->" + axis
        mask = tf.cast(tf.einsum(subscripts, *U), dtype=tf.bool)
        return(mask)

    def testMask(self, Mf, foldNumber, nFolds, nValues):
        Uf = np.zeros(Mf)
        if foldNumber == nFolds-1:
            Uf[(nFolds-1)*nValues:] = 1.
        else:
            Uf[foldNumber*nValues:(foldNumber+1)*nValues] = 1.
        return(Uf)
