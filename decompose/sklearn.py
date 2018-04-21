from typing import Tuple, List
import numpy as np
import string
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from decompose.models.tensorFactorisation import TensorFactorisation
from decompose.distributions.cenNormal import CenNormal
from decompose.stopCriterions.stopCriterion import StopHook
from decompose.stopCriterions.llhImprovementThreshold import LlhImprovementThreshold
from decompose.stopCriterions.llhStall import LlhStall


class DECOMPOSE(object):
    """An interface to DECMPOSE similar to sklearn.decompose."""

    def __init__(self, modelDirectory: str,
                 priors: Tuple[type, ...] = (CenNormal, CenNormal),
                 n_components: int = 3,
                 dtype: type = np.float32,
                 maxIterations: int = 100000,
                 doRescale: bool = True,
                 stopCriterionInit=LlhStall(10, ns="scInit"),
                 stopCriterionEM=LlhStall(100, ns="sc0"),
                 stopCriterionBCD=LlhImprovementThreshold(1e-2, ns="sc1"),
                 device: str = "/cpu:0") -> None:
        self.__maxIterations = maxIterations
        self.__n_components = n_components
        self.__priors = priors
        self.__dtype = dtype
        self.__modelDirectory = modelDirectory
        self.__device = device
        self.__doRescale = doRescale
        tefa = TensorFactorisation.getEstimator(
            priors=priors,
            K=self.n_components,
            dtype=tf.as_dtype(dtype),
            path=modelDirectory,
            doRescale=doRescale,
            stopCriterionInit=stopCriterionInit,
            stopCriterionEM=stopCriterionEM,
            stopCriterionBCD=stopCriterionBCD,
            device=self.__device)
        self.__tefa = tefa

    @property
    def doRescale(self):
        return(self.__doRescale)

    @property
    def n_components(self):
        return(self.__n_components)

    @property
    def components_(self):
        U = self.__components_
        if len(U) == 1:
            return(U[0])
        else:
            return(self.__components_)

    @property
    def variance_ratio_(self):
        return(self.__variance_ratio)

    def __calc_variance_ratio(self, data, U):
        varData = np.var(data)
        evr = np.zeros(self.n_components)
        F = len(U)
        axisIds = string.ascii_lowercase[:F]
        subscripts = f'k{",k".join(axisIds)}->{axisIds}'
        for k in range(self.n_components):
            Uks = []
            for Uf in U:
                Uks.append(Uf[k][None, ...])
            recons = np.einsum(subscripts, *Uks)
            evr[k] = np.var(recons)/varData
        return(evr)

    def fit(self, X: np.ndarray):
        # create input_fn
        x = {"train": X.astype(self.__dtype)}
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x, y=None, batch_size=X.shape[0],
            shuffle=False, num_epochs=None, )

        # train the model
        self.__tefa.train(input_fn=input_fn,
                          steps=self.__maxIterations,
                          hooks=[StopHook()])

        # store result
        ckptFile = self.__tefa.latest_checkpoint()
        ckptReader = pywrap_tensorflow.NewCheckpointReader(ckptFile)
        UsList = []  # type: List[tf.Tensor]
        F = len(X.shape)
        for f in range(F):
            Uf = ckptReader.get_tensor(f"U/{f}")
            UsList.append(Uf)
        Us = tuple(UsList)
        self.__variance_ratio = self.__calc_variance_ratio(X, Us)
        self.__components_ = Us[1:]
        return(self)

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        ckptFile = self.__tefa.latest_checkpoint()
        ckptReader = pywrap_tensorflow.NewCheckpointReader(ckptFile)
        U0 = ckptReader.get_tensor("U/0")
        return(U0)

    def transform(self, X: np.ndarray, transformModelDirectory: str):
        # create input_fn
        x = {"test": X}
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x, y=None, batch_size=X.shape[0],
            shuffle=False, num_epochs=None)

        ckptFile = self.__tefa.latest_checkpoint()
        tefaTransform = TensorFactorisation.getTransformEstimator(
            priors=self.__priors,
            K=self.n_components,
            dtype=self.__dtype,
            path=transformModelDirectory,
            chptFile=ckptFile)
        tefaTransform.train(input_fn=input_fn,
                            steps=self.__maxIterations,
                            hooks=[StopHook()])
        ckptFile = tefaTransform.latest_checkpoint()
        ckptReader = pywrap_tensorflow.NewCheckpointReader(ckptFile)
        U0 = ckptReader.get_tensor("U/0tr")
        return(U0)
