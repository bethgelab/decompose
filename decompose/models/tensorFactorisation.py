from typing import Tuple, List, Dict, Any
from enum import Enum
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow import Tensor
from copy import copy

from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.distribution import Distribution
from decompose.distributions.uniform import Uniform
from decompose.distributions.nnUniform import NnUniform
from decompose.likelihoods.likelihood import Likelihood
from decompose.likelihoods.specificNormal2dLikelihood import SpecificNormal2dLikelihood
from decompose.likelihoods.allSpecificNormal2dLikelihood import AllSpecificNormal2dLikelihood
from decompose.likelihoods.normal2dLikelihood import Normal2dLikelihood
from decompose.likelihoods.normalNdLikelihood import NormalNdLikelihood
from decompose.likelihoods.cvNormal2dLikelihood import CVNormal2dLikelihood
from decompose.likelihoods.cvNormalNdLikelihood import CVNormalNdLikelihood
from decompose.postU.postU import PostU
from decompose.stopCriterions.llhImprovementThreshold import LlhImprovementThreshold
from decompose.stopCriterions.llhStall import LlhStall
from decompose.cv.cv import CV


EstimatorSpec = tf.estimator.EstimatorSpec


class NoiseUniformity(Enum):
    HOMOGENEOUS = 0
    HETEROGENEOUS = 1
    LAST_FACTOR_HETEROGENOUS = 2


HOMOGENEOUS = NoiseUniformity.HOMOGENEOUS
HETEROGENEOUS = NoiseUniformity.HETEROGENEOUS
LAST_FACTOR_HETEROGENOUS = NoiseUniformity.LAST_FACTOR_HETEROGENOUS


class parameterProperty(object):
    """Decorator for descriptors that update tf variables during set.

    This decorator is the same as the python property decorator except
    that its setter method accepts a name which can updates a
    tensorflow variable depending on
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None, name=None):
        self.name = name
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __set__(self, obj, values):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        newValues = []
        for f, value in enumerate(values):
            if obj.transform and (f == 0):
                name = f"{f}tr"
            else:
                name = f"{f}"
            with tf.variable_scope("U", reuse=tf.AUTO_REUSE):
                var = tf.get_variable(name, dtype=obj.dtype)
            value = tf.assign(var, value)
            newValues.append(value)
        value = tuple(newValues)
        self.fset(obj, value)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__, self.name)

    def setter(self, name):
        if name is None:
            return type(self)(self.fget, None, self.fdel, self.__doc__, None)
        if type(name) is not str:
            raise ValueError("setter takes a name argument as a string")

        def noop(fset):
            return type(self)(self.fget, fset, self.fdel, self.__doc__, name)
        return(noop)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__, self.name)


class Phase(Enum):
    INIT = 1
    EM = 2
    BCD = 3


class TensorFactorisation(object):

    def __init__(self,
                 U: List[Tensor],
                 priorU: List[Distribution],
                 likelihood: Likelihood,
                 dtype: tf.DType,
                 stopCriterion,
                 phase: Phase,
                 noiseUniformity: NoiseUniformity,
                 transform: bool = False) -> None:

        # setup the model
        self.dtype = dtype
        self.__transform = transform
        self.__noiseUniformity = noiseUniformity
        self.likelihood = likelihood
        self.stopCriterion = stopCriterion
        self.postU = []  # type: List[PostU]
        for f, priorUf in enumerate(priorU):
            postUf = PostU(likelihood, priorUf, f)
            self.postU.append(postUf)

        # create or reuse the variables for the filter banks
        for f, Uf in enumerate(copy(U)):
            if transform and (f == 0):
                paramName = "{}tr".format(f)
            else:
                paramName = "{}".format(f)
            with tf.variable_scope("U", reuse=tf.AUTO_REUSE):
                UfVar = tf.get_variable(paramName,
                                        dtype=dtype,
                                        initializer=Uf)
            U[f] = UfVar
        self.__U = tuple(U)
        if phase == Phase.EM or phase == Phase.INIT:
            self.__setEm()
        elif phase == Phase.BCD:
            self.__setBcd()
        else:
            raise ValueError

    @classmethod
    def random(cls,
               priorU: List[Distribution],
               likelihood: Likelihood,
               M: Tuple[int, ...],
               K: int,
               dtype: tf.DType,
               phase: Phase,
               stopCriterion,
               noiseUniformity: NoiseUniformity = HOMOGENEOUS,
               transform: bool = False) -> "TensorFactorisation":

        # initialize U
        dtype = tf.as_dtype(dtype)
        zero = tf.constant(0., dtype=dtype)
        one = tf.constant(1., dtype=dtype)
        normal = tf.distributions.Normal(loc=zero, scale=one)
        F = len(M)
        U = []
        for f in range(F):
            if priorU[f].nonNegative:
                UfInit = tf.abs(normal.sample(sample_shape=(K, M[f])))
            else:
                UfInit = normal.sample(sample_shape=(K, M[f]))
            U.append(UfInit)

        # instantiate
        tefa = TensorFactorisation(U=U,
                                   priorU=priorU,
                                   likelihood=likelihood,
                                   dtype=dtype,
                                   phase=phase,
                                   transform=transform,
                                   noiseUniformity=noiseUniformity,
                                   stopCriterion=stopCriterion)
        return(tefa)

    @property
    def transform(self) -> bool:
        return(self.__transform)

    @property
    def noiseUniformity(self) -> NoiseUniformity:
        return(self.__noiseUniformity)

    @parameterProperty
    def U(self) -> Tuple[tf.Tensor, ...]:
        return(self.__U)

    @U.setter(name="U")
    def U(self, U: Tuple[tf.Tensor, ...]):
        self.__U = U

    def update(self, X: Tensor) -> Tuple[Tensor, ...]:
        # update stopping criterion
        stopCritDeps = self.stopCriterion.update(self, X)

        # perform updated depending on this is train of transformation
        with tf.control_dependencies(stopCritDeps):
            if self.transform:
                self.updateTransform(X)
            else:
                self.updateTrain(X)

        # return the updated tensors
        return(self.U)

    def updateTrain(self, X: Tensor) -> None:
        # store filterbanks in a list
        U = list(self.U)  # type: List[Tensor]

        # update the parameters of the likelihood
        self.likelihood.update(U=U, X=X)

        # update the filters in reversed order
        for f, postUf in reversed(list(enumerate(self.postU))):
            U = self.rescale(U=U, fNonUnit=f)
            U[f] = postUf.update(U=U, X=X, transform=False)

        # update the filter banks
        self.U = tuple(U)

    def updateTransform(self, X: Tensor) -> None:
        # store filterbanks in a list
        U = list(self.U)  # type: List[Tensor]

        # calculate updates of the first filterbank
        U[0] = self.postU[0].update(U=U, X=X, transform=True)

        # update the filter banks
        self.U = tuple(U)

    def rescale(self, U: List[Tensor], fNonUnit: int) -> List[Tensor]:
        """Puts all variance in the factor `fUpdate`-th factor.

        The method assumes that the norm of all filters is larger than 0."""
        F = len(U)

        # calculathe the scale for each source
        scaleOfSources = tf.ones_like(U[0][..., 0])
        for f in range(F):
            scaleOfSources = scaleOfSources*tf.norm(U[f], axis=-1)

        for f in range(F):
            # determine rescaling constant depending on the factor number
            Uf = U[f]
            normUf = tf.norm(Uf, axis=-1)
            if f == fNonUnit:
                # put all variance in the filters of the fUpdate-th factor
                rescaleConstant = scaleOfSources/normUf
            else:
                # normalize the filters all other factors
                rescaleConstant = 1./normUf

            # rescaled filters
            Uf = Uf*rescaleConstant[..., None]
            U[f] = Uf

        return(U)

    def __setEm(self) -> None:
        """Set prior and noise distributions to perform EM updates."""
        for postUf in self.postU:
            postUf.prior.drawType = DrawType.SAMPLE
            postUf.prior.updateType = UpdateType.ALL
        self.likelihood.noiseDistribution.drawType = DrawType.SAMPLE
        self.likelihood.noiseDistribution.updateType = UpdateType.ALL

    def __setBcd(self) -> None:
        """Set prior and noise distributions to perform BCD updates."""
        for postUf in self.postU:
            postUf.prior.drawType = DrawType.MODE
            postUf.prior.updateType = UpdateType.ONLYLATENTS
        self.likelihood.noiseDistribution.drawType = DrawType.MODE
        self.likelihood.noiseDistribution.updateType = UpdateType.ONLYLATENTS

    def loss(self, X: Tensor) -> Tensor:
        """Loss of the data `X` given the parameters."""
        loss = self.likelihood.loss(self.U, X)
        loss = tf.cast(loss, tf.float64)
        return(loss)

    def llh(self, X: Tensor) -> Tensor:
        """Log likelihood of the parameters given data `X`."""

        # log likelihood of the noise
        llh = self.likelihood.llh(self.U, X)

        # log likelihood of the factors
        U = list(self.U)
        for f, postUf in enumerate(self.postU):
            U = self.rescale(U=U, fNonUnit=f)
            UfT = tf.transpose(U[f])
            llhUf = tf.reduce_sum(postUf.prior.llh(UfT))
            llh = llh + llhUf
        llh = tf.cast(llh, tf.float64)
        return(llh)

    def llhIndividual(self, X: Tensor) -> Tensor:
        """Log likelihood of the parameters given data `X`."""

        # log likelihood of the noise
        llhRes = self.likelihood.llh(self.U, X)
        llh = llhRes

        # log likelihood of the factors
        llhU = []
        llhUfk = []
        U = list(self.U)
        for f, postUf in enumerate(self.postU):
            U = self.rescale(U=U, fNonUnit=f)
            UfT = tf.transpose(U[f])
            llhUfk.append(tf.reduce_sum(postUf.prior.llh(UfT), axis=0))
            llhUf = tf.reduce_sum(postUf.prior.llh(UfT))
            llh = llh + llhUf
            llhU.append(llhUf)
        llh = tf.cast(llh, tf.float64)
        return(llh, llhRes, llhU, llhUfk)

    @staticmethod
    def type():
        return(TensorFactorisation)

    def id(self) -> str:
        """Generate a string representation of the model configuration"""
        strId = ""
        for f, postUf in enumerate(self.postU):
            strId += "U{}".format(f) + postUf.prior.id()
        strId += "_" + self.likelihood.id
        return(strId)

    @classmethod
    def __model(cls, data: Tensor, priorTypes: List[Distribution],
                M: Tuple[int, ...],
                K: int, stopCriterion, phase: Phase, dtype: tf.DType,
                reuse=False,
                isFullyObserved: bool = True,
                cv: CV = None,
                transform: bool = False,
                noiseUniformity: NoiseUniformity = HOMOGENEOUS,
                suffix: str = "") -> "TensorFactorisation":
        varscope = "stopCriterion" + phase.name
        stopCriterion.init(ns=varscope)
        F = len(priorTypes)

        # selecting the apropriate likelihood
        useNormal2dLikelihood = (
            F == 2
            and cv is None
            and isFullyObserved
            and noiseUniformity == HOMOGENEOUS)
        useAllSpecificNormal2dLikelihood = (
            F == 2
            and cv is None
            and isFullyObserved
            and noiseUniformity == HETEROGENEOUS)
        useSpecificNormal2dLikelihood = (
            F == 2
            and cv is None
            and isFullyObserved
            and noiseUniformity == LAST_FACTOR_HETEROGENOUS)
        useCVNormal2dLikelihood = (
            F == 2
            and (cv is not None
                 or not isFullyObserved)
            and noiseUniformity == HOMOGENEOUS)
        useNormalNdLikelihood = (
            F > 2
            and cv is None
            and isFullyObserved
            and noiseUniformity == HOMOGENEOUS)
        useCVNormalNdLikelihood = (
            F > 2
            and (cv is not None
                 or not isFullyObserved)
            and noiseUniformity == HOMOGENEOUS)

        # instantiate the likelihood
        with tf.variable_scope("", reuse=reuse):
            if useNormal2dLikelihood:
                likelihood = Normal2dLikelihood(
                    M=M, K=K, dtype=dtype)  # type: Likelihood
            elif useAllSpecificNormal2dLikelihood:
                likelihood = AllSpecificNormal2dLikelihood(
                    M=M, K=K, dtype=dtype)
            elif useSpecificNormal2dLikelihood:
                likelihood = SpecificNormal2dLikelihood(
                    M=M, K=K, dtype=dtype)
            elif useCVNormal2dLikelihood:
                likelihood = CVNormal2dLikelihood(
                    M=M, K=K, dtype=dtype, cv=cv)
            elif useNormalNdLikelihood:
                likelihood = NormalNdLikelihood(
                    M=M, K=K, dtype=dtype)
            elif useCVNormalNdLikelihood:
                likelihood = CVNormalNdLikelihood(
                    M=M, K=K, dtype=dtype, cv=cv)
            else:
                raise NotImplementedError()
            likelihood.init(data)

            # instantiate the priors
            priors = []
            for f, priorType in enumerate(priorTypes):
                prior = priorType.random(shape=(K,), latentShape=(M[f],),
                                         name=f"prior{suffix}{f}", dtype=dtype)
                priors.append(prior)

        # instantiate the model
        tefa = cls.random(priorU=priors, likelihood=likelihood, M=M, K=K,
                          phase=phase, stopCriterion=stopCriterion,
                          dtype=dtype, noiseUniformity=noiseUniformity,
                          transform=transform)
        return(tefa)

    @classmethod
    def __estimatorSpec(cls, mode, features, device: str,
                        isFullyObserved: bool,
                        priors: List[Distribution],
                        K: int, stopCriterionInit, stopCriterionEM,
                        stopCriterionBCD,
                        cv: CV, path: str,
                        noiseUniformity: NoiseUniformity,
                        transform: bool, dtype: tf.DType) -> EstimatorSpec:
        # PREDICT and EVAL are not supported
        if mode != tf.estimator.ModeKeys.TRAIN:
            raise ValueError

        # TRAIN
        with tf.device(device):
            # check the input data
            labels = list(features.keys())
            assert len(labels) == 1
            data = features[labels[0]]
            dataShape = tuple(data.get_shape().as_list())
            assert len(dataShape) == len(priors)

            # shape of the data
            M = data.get_shape().as_list()

            # create llh variable
            inf = np.float64(np.inf)
            with tf.variable_scope("llh"):
                llhVar = tf.get_variable("llh", dtype=tf.float64,
                                         initializer=-inf)

            # create loss variable
            with tf.variable_scope("loss"):
                lossVar = tf.get_variable("loss", dtype=tf.float64,
                                          initializer=inf)

            # create global stopping variable
            with tf.variable_scope("stopCriterion"):
                stopVar = tf.get_variable("stop", dtype=tf.bool,
                                          initializer=False)

            # INIT model
            initPriors = []  # type: List[Distribution]
            for prior in priors:
                if prior.nonNegative:
                    initPriors.append(NnUniform())
                else:
                    initPriors.append(Uniform())
            tefaInit = cls.__model(data=data, priorTypes=initPriors, K=K, M=M,
                                   isFullyObserved=isFullyObserved,
                                   stopCriterion=stopCriterionInit,
                                   dtype=dtype, reuse=False,
                                   transform=transform, cv=cv,
                                   phase=Phase.INIT,
                                   noiseUniformity=noiseUniformity,
                                   suffix="init")

            # EM model
            tefaEM = cls.__model(data=data, priorTypes=priors, K=K, M=M,
                                 isFullyObserved=isFullyObserved,
                                 stopCriterion=stopCriterionEM,
                                 dtype=dtype, phase=Phase.EM,
                                 transform=transform, cv=cv,
                                 noiseUniformity=noiseUniformity,
                                 reuse=tf.AUTO_REUSE)

            # BCD model
            tefaBCD = cls.__model(data=data, priorTypes=priors, K=K, M=M,
                                  isFullyObserved=isFullyObserved,
                                  stopCriterion=stopCriterionBCD,
                                  dtype=dtype, phase=Phase.BCD,
                                  transform=transform, cv=cv,
                                  noiseUniformity=noiseUniformity,
                                  reuse=tf.AUTO_REUSE)

            # replace nan with zeros
            data = tf.where(tf.is_nan(data), tf.zeros_like(data), data)

            # conduct an update depending on the current phase
            stopVarInit = tefaInit.stopCriterion.stopVar
            stopVarEm = tefaEM.stopCriterion.stopVar
            loss = tf.cond(tf.logical_not(stopVarInit),
                           lambda: tefaInit.loss(X=data),
                           lambda: tf.cond(tf.logical_not(stopVarEm),
                                           lambda: tefaEM.loss(X=data),
                                           lambda: tefaBCD.loss(X=data)))

            # conduct an update depending on the current phase
            deps = tf.cond(tf.logical_not(stopVarInit),
                           lambda: tefaInit.update(X=data),
                           lambda: tf.cond(tf.logical_not(stopVarEm),
                                           lambda: tefaEM.update(X=data),
                                           lambda: tefaBCD.update(X=data)))

            # update the global stop variable
            stopVarBcd = tefaBCD.stopCriterion.stopVar
            stop = tf.logical_and(stopVarInit,
                                  tf.logical_and(stopVarEm,
                                                 stopVarBcd))
            with tf.control_dependencies(deps):
                updatedStopVar = tf.assign(stopVar, stop)

            # if stopping criterion is reached store the llh
            updates = tf.cond(stop,
                              lambda: (tf.assign(llhVar, tefaBCD.llh(data)),
                                       tf.assign(lossVar, tefaBCD.loss(data))),
                              lambda: (llhVar, lossVar))

            # increment global step variable
            with tf.control_dependencies([updatedStopVar, *updates]):
                step = tf.train.get_or_create_global_step()
                trainOp = tf.assign(step, step + 1)

            # log summaries
            tf.summary.scalar("loss", loss)
            llh = tf.cond(tf.logical_not(stopVarInit),
                          lambda: tefaInit.llhIndividual(X=data),
                          lambda: tf.cond(tf.logical_not(stopVarEm),
                                          lambda: tefaEM.llhIndividual(X=data),
                                          lambda: tefaBCD.llhIndividual(X=data)))
            llh, llhRes, llhU, llhUfk = llh
            tf.summary.scalar("llh", llh)
            tf.summary.scalar("llhResiduals", llhRes)
            for f, (llhUf, llhUfk) in enumerate(zip(llhU, llhUfk)):
                tf.summary.scalar(f"llhU{f}", llhUf)

            SAVE_EVERY_N_STEPS = 1  # TODO: make configurable
            summary_hook = tf.train.SummarySaverHook(
                SAVE_EVERY_N_STEPS,
                output_dir=path,
                summary_op=tf.summary.merge_all())

        return EstimatorSpec(mode, loss=loss, train_op=trainOp,
                             training_hooks=[summary_hook])

    @classmethod
    def getEstimator(cls, priors: Tuple[Distribution, ...], K: int,
                     dtype: tf.DType = tf.float32,
                     isFullyObserved: bool = True,
                     noiseUniformity: NoiseUniformity = HOMOGENEOUS,
                     stopCriterionInit=LlhStall(10),
                     stopCriterionEM=LlhStall(100),
                     stopCriterionBCD=LlhImprovementThreshold(1e-2),
                     path: str = "/tmp", device: str = "/cpu:0",
                     cv: CV = None):

        def model_fn(features, labels, mode):
            es = cls.__estimatorSpec(mode=mode, features=features,
                                     isFullyObserved=isFullyObserved,
                                     device=device, priors=priors,
                                     noiseUniformity=noiseUniformity,
                                     stopCriterionInit=stopCriterionInit,
                                     stopCriterionEM=stopCriterionEM,
                                     stopCriterionBCD=stopCriterionBCD,
                                     cv=cv, path=path, K=K,
                                     transform=False, dtype=dtype)
            return(es)

        est = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=path)
        return(est)

    @classmethod
    def getTransformEstimator(cls, priors: Tuple[Distribution, ...], K: int,
                              chptFile: str, dtype: tf.DType = tf.float32,
                              noiseUniformity: NoiseUniformity = HOMOGENEOUS,
                              stopCriterionInit=LlhStall(10),
                              stopCriterionEM=LlhStall(100),
                              stopCriterionBCD=LlhImprovementThreshold(1e-2),
                              path: str = "/tmp", device: str = "/cpu:0"):
        # configuring warm start settings
        reader = pywrap_tensorflow.NewCheckpointReader(chptFile)
        varList = [v for v in reader.get_variable_to_shape_map().keys()
                   if (v != "U/0" and
                       v != "global_step" and
                       v != "stop" and
                       not v.startswith(f"stopCriterion{Phase.INIT.name}/") and
                       not v.startswith(f"stopCriterion{Phase.EM.name}/") and
                       not v.startswith(f"stopCriterion{Phase.BCD.name}/"))]
        wsVars = "|".join(varList)
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=chptFile,
                                            vars_to_warm_start=wsVars)

        def model_fn(features, labels, mode):
            es = cls.__estimatorSpec(mode=mode, features=features,
                                     isFullyObserved=True,
                                     device=device, priors=priors,
                                     noiseUniformity=noiseUniformity,
                                     stopCriterionInit=stopCriterionInit,
                                     stopCriterionEM=stopCriterionEM,
                                     stopCriterionBCD=stopCriterionBCD,
                                     K=K, path=path, cv=None,
                                     transform=True, dtype=dtype)
            return(es)

        est = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=path,
                                     warm_start_from=ws)
        return(est)
