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
from decompose.likelihoods.likelihood import Likelihood
from decompose.likelihoods.normal2dLikelihood import Normal2dLikelihood
from decompose.postU.postU import PostU
from decompose.stopCriterions.stopCriterion import NoStop
from decompose.stopCriterions.llhImprovementThreshold import LlhImprovementThreshold
from decompose.stopCriterions.llhStall import LlhStall
from decompose.stopCriterions.nIterations import NIterations


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
    EM = 1
    BCD = 2


class TensorFactorisation(object):

    def __init__(self,
                 U: List[Tensor],
                 priorU: List[Distribution],
                 likelihood: Likelihood,
                 dtype,
                 stopCriterion,
                 phase,
                 doRescale: bool = True,
                 transform: bool = False) -> None:

        # setup the model
        self.dtype = dtype
        self.__doRescale = doRescale
        self.__transform = transform
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
        if phase == Phase.EM:
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
               dtype,
               phase,
               stopCriterion,
               doRescale: bool = True,
               transform: bool = False) -> "TensorFactorisation":

        # initialize U
        U = [tf.constant(np.random.normal(size=(K, M[0])), dtype=dtype),
             tf.constant(np.random.normal(size=(K, M[1])), dtype=dtype)]

        # instantiate
        tefa = TensorFactorisation(U=U,
                                   priorU=priorU,
                                   likelihood=likelihood,
                                   dtype=dtype,
                                   phase=phase,
                                   transform=transform,
                                   doRescale=doRescale,
                                   stopCriterion=stopCriterion)
        return(tefa)

    @property
    def doRescale(self) -> bool:
        return(self.__doRescale)

    @property
    def transform(self) -> bool:
        return(self.__transform)

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

    def updateTrain(self, X: Tensor):
            # store filterbanks in a list
            U = list(self.U)  # type: List[Tensor]

            # update the parameters of the likelihood
            self.likelihood.update(U, X)

            # calculate updates of the filterbanks
            for f, postUf in enumerate(self.postU):
                U[f] = postUf.update(U, X, False)

                if self.doRescale:
                    U = self.rescale(U)

            # update the filter banks
            self.U = tuple(U)

    def updateTransform(self, X: Tensor):
            # store filterbanks in a list
            U = list(self.U)  # type: List[Tensor]

            # calculate updates of the last filterbank
            U[0] = self.postU[0].update(U, X, True)

            # update the filter banks
            self.U = tuple(U)

    def rescale(self, U: List[Tensor]) -> List[Tensor]:
        """Same l2 norm across all factors."""

        F = len(U)
        norm = tf.ones_like(U[0][..., 0])

        for f in range(F):
            norm = norm*tf.norm(U[f], axis=-1)
        norm = (norm)**(1./F)

        for f in range(F):
            Uf = U[f]
            normUf = tf.norm(Uf, axis=-1)
            norm = tf.where(tf.logical_or(tf.equal(norm/normUf, 0.),
                                          tf.logical_not(tf.is_finite(norm/normUf))),
                            tf.zeros_like(norm), norm)
        for f in range(F):
            Uf = U[f]
            normUf = tf.norm(Uf, axis=-1)
            Uf = Uf*(norm/normUf)[..., None]
            Uf = tf.where(tf.logical_or(tf.equal(norm/normUf, 0.),
                                        tf.logical_not(tf.is_finite(norm/normUf))), U[f], Uf)
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

    def residuals(self, X: Tensor):
        """Difference between the data and its reconstruction"""
        r = self.likelihood.residuals(list(self.U), X)
        return(r)

    def llh(self, X: Tensor) -> float:
        """Evaluates the log likelihood of the model"""

        # log likelihood of the noise
        r = self.residuals(X)
        noiseDistribution = self.likelihood.noiseDistribution
        llh = tf.reduce_sum(noiseDistribution.llh(tf.reshape(r, (-1,))))

        # log likelihood of the factors
        for f, postUf in enumerate(self.postU):
            llhUf = tf.reduce_sum(postUf.prior.llh(tf.transpose(self.U[f])))
            llh = llh + llhUf
        return(llh)

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

    @staticmethod
    def getSinglePhaseEstimator(priors, K: int, dtype, phase=Phase.EM,
                                stopCriterion=NoStop(), path: str = "/tmp"):

        def model_fn(features, labels, mode):
            # PREDICT and EVAL are not supported
            if mode != tf.estimator.ModeKeys.TRAIN:
                raise ValueError

            # TRAIN
            labels = list(features.keys())
            assert len(labels) == 1
            data = features[labels[0]]
            dtype = data.dtype
            dataShape = tuple(data.get_shape().as_list())
            assert len(dataShape) == len(priors)
            M = data.get_shape().as_list()

            stopCriterion.init()
            likelihood = Normal2dLikelihood(M=M, K=K, dtype=dtype)
            priorDists = [prior.random(shape=(K,),
                                       latentShape=(M[f],),
                                       name=f"prior{f}",
                                       dtype=dtype)
                          for f, prior in enumerate(priors)]

            tefa = TensorFactorisation.random(priorU=priorDists,
                                              likelihood=likelihood,
                                              M=M,
                                              K=K,
                                              phase=phase,
                                              stopCriterion=stopCriterion,
                                              dtype=dtype)
            loss = tf.reduce_sum(tefa.residuals(data)**2)
            updatedU = tefa.update(X=data)

            with tf.control_dependencies(updatedU):
                step = tf.train.get_or_create_global_step()
                trainOp = tf.assign(step, step + 1)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=trainOp)

        est = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=path)
        return(est)

    @staticmethod
    def getEstimator(priors, K: int, dtype,
                     stopCriterion=(LlhStall(100, ns="sc0"),
                                    LlhImprovementThreshold(1e-2, ns="sc1")),
                     path: str = "/tmp",
                     device: str = "/cpu:0",
                     doRescale: bool = True):

        def model_fn(features, labels, mode):

            # PREDICT and EVAL are not supported
            if mode != tf.estimator.ModeKeys.TRAIN:
                raise ValueError

            # TRAIN
            with tf.device(device):
                with tf.variable_scope("stopCriterion"):
                    stopVar = tf.get_variable("stop",
                                              dtype=tf.bool,
                                              initializer=False)
                labels = list(features.keys())
                assert len(labels) == 1
                data = features[labels[0]]
                dtype = data.dtype
                dataShape = tuple(data.get_shape().as_list())
                assert len(dataShape) == len(priors)
                M = data.get_shape().as_list()

                # INIT model
                stopCriterionInit = NIterations(100, ns="scInit")
                stopCriterionInit.init()
                priorDistsInit = [Uniform.random(shape=(K,),
                                                 latentShape=(M[f],),
                                                 name=f"priorInit{f}",
                                                 dtype=dtype)
                                  for f, prior in enumerate(priors)]
                likelihood = Normal2dLikelihood(M=M, K=K, dtype=dtype)
                tefaInit = TensorFactorisation.random(priorU=priorDistsInit,
                                                      likelihood=likelihood,
                                                      M=M,
                                                      K=K,
                                                      phase=Phase.EM,
                                                      stopCriterion=stopCriterionInit,
                                                      doRescale=doRescale,
                                                      dtype=dtype)

                # EM model
                stopCriterion[0].init()
                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    likelihood = Normal2dLikelihood(M=M, K=K, dtype=dtype)
                priorDistsEm = [prior.random(shape=(K,),
                                             latentShape=(M[f],),
                                             name=f"prior{f}",
                                             dtype=dtype)
                                for f, prior in enumerate(priors)]
                tefaEm = TensorFactorisation.random(priorU=priorDistsEm,
                                                    likelihood=likelihood,
                                                    M=M,
                                                    K=K,
                                                    phase=Phase.EM,
                                                    stopCriterion=stopCriterion[0],
                                                    doRescale=doRescale,
                                                    dtype=dtype)

                # BCD model
                stopCriterion[1].init()
                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    likelihoodBcd = Normal2dLikelihood(M=M, K=K, dtype=dtype)
                    priorDistsBcd = [prior.random(shape=(K,),
                                                  latentShape=(M[f],),
                                                  name=f"prior{f}",
                                                  dtype=dtype)
                                     for f, prior in enumerate(priors)]

                tefaBcd = TensorFactorisation.random(priorU=priorDistsBcd,
                                                     likelihood=likelihoodBcd,
                                                     M=M,
                                                     K=K,
                                                     phase=Phase.BCD,
                                                     stopCriterion=stopCriterion[1],
                                                     doRescale=doRescale,
                                                     dtype=dtype)
                loss = tf.reduce_sum(tefaEm.residuals(data)**2)

                # conduct the updated depending on the stop criteria
                updatedU = tf.cond(tf.logical_not(tefaInit.stopCriterion.stopVar),
                                   lambda: tefaInit.update(X=data),
                                   lambda: tf.cond(tf.logical_not(tefaEm.stopCriterion.stopVar),
                                                   lambda: tefaEm.update(X=data),
                                                   lambda: tefaBcd.update(X=data)))

                # update the global stop variable
                with tf.control_dependencies(updatedU):
                    updatedStopVar = tf.assign(
                        stopVar,
                        tf.logical_and(tefaInit.stopCriterion.stopVar,
                                       tf.logical_and(tefaEm.stopCriterion.stopVar,
                                                      tefaBcd.stopCriterion.stopVar)))

                with tf.control_dependencies([updatedStopVar]):
                    step = tf.train.get_or_create_global_step()
                    trainOp = tf.assign(step, step + 1)
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=trainOp)

        est = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=path)
        return(est)

    @staticmethod
    def getSinglePhaseTransformEstimator(priors, K: int, dtype, chptFile: str,
                                         phase=Phase.EM, stopCriterion=NoStop(),
                                         path: str = "/tmp"):

        reader = pywrap_tensorflow.NewCheckpointReader(chptFile)
        varList = [v for v in reader.get_variable_to_shape_map().keys()
                   if v != "U/0" and v != "global_step"]

        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=chptFile,
                                            vars_to_warm_start="|".join(varList))

        def model_fn(features, labels, mode):
            # PREDICT and EVAL are not supported
            if mode != tf.estimator.ModeKeys.TRAIN:
                raise ValueError

            # TRAIN
            labels = list(features.keys())
            assert len(labels) == 1
            data = features[labels[0]]
            dtype = data.dtype
            dataShape = tuple(data.get_shape().as_list())

            assert len(dataShape) == len(priors)

            stopCriterion.init()

            M = data.get_shape().as_list()

            likelihood = Normal2dLikelihood(M=M, K=K, dtype=dtype)
            priorDists = [prior.random(shape=(K,),
                                       latentShape=(M[f],),
                                       name="prior{}".format(f),
                                       dtype=dtype)
                          for f, prior in enumerate(priors)]

            tefa = TensorFactorisation.random(priorU=priorDists,
                                              likelihood=likelihood,
                                              M=M,
                                              K=K,
                                              phase=phase,
                                              dtype=dtype,
                                              stopCriterion=stopCriterion,
                                              transform=True)
            loss = tf.reduce_sum(tefa.residuals(data)**2)
            updatedU = tefa.update(X=data)

            with tf.control_dependencies(updatedU):
                step = tf.train.get_or_create_global_step()
                train_op_normal = tf.assign(step, step + 1)
            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=train_op_normal)

        est = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=path,
                                     warm_start_from=ws)

        return(est)

    @staticmethod
    def getTransformEstimator(priors, K: int, dtype, chptFile: str,
                              stopCriterion=(NIterations(10, ns="sc0"),
                                             NIterations(10, ns="sc1")),
                              path: str = "/tmp"):

        reader = pywrap_tensorflow.NewCheckpointReader(chptFile)
        varList = [v for v in reader.get_variable_to_shape_map().keys()
                   if (v != "U/0" and
                       v != "global_step" and
                       v != "stop" and
                       not v.startswith("sc0/") and
                       not v.startswith("sc1/"))]

        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=chptFile,
                                            vars_to_warm_start="|".join(varList))

        def model_fn(features, labels, mode):
            # PREDICT and EVAL are not supported
            if mode != tf.estimator.ModeKeys.TRAIN:
                raise ValueError

            # TRAIN
            with tf.variable_scope("stopCriterion"):
                stopVar = tf.get_variable("stop",
                                          dtype=tf.bool,
                                          initializer=False)
            labels = list(features.keys())
            assert len(labels) == 1
            data = features[labels[0]]
            dtype = data.dtype
            dataShape = tuple(data.get_shape().as_list())

            assert len(dataShape) == len(priors)

            M = data.get_shape().as_list()

            # EM model
            stopCriterion[0].init()
            likelihoodEm = Normal2dLikelihood(M=M, K=K, dtype=dtype)
            priorDistsEm = [prior.random(shape=(K,),
                                         latentShape=(M[f],),
                                         name=f"prior{f}",
                                         dtype=dtype)
                            for f, prior in enumerate(priors)]
            tefaEm = TensorFactorisation.random(priorU=priorDistsEm,
                                                likelihood=likelihoodEm,
                                                M=M,
                                                K=K,
                                                phase=Phase.EM,
                                                stopCriterion=stopCriterion[0],
                                                dtype=dtype,
                                                transform=True)

            # BCD model
            stopCriterion[1].init()
            with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                likelihoodBcd = Normal2dLikelihood(M=M, K=K, dtype=dtype)
                priorDistsBcd = [prior.random(shape=(K,),
                                              latentShape=(M[f],),
                                              name=f"prior{f}",
                                              dtype=dtype)
                                 for f, prior in enumerate(priors)]

            tefaBcd = TensorFactorisation.random(priorU=priorDistsBcd,
                                                 likelihood=likelihoodBcd,
                                                 M=M,
                                                 K=K,
                                                 phase=Phase.BCD,
                                                 stopCriterion=stopCriterion[1],
                                                 dtype=dtype,
                                                 transform=True)
            loss = tf.reduce_sum(tefaEm.residuals(data)**2)

            # conduct the updated depending on the stop criteria
            updatedU = tf.cond(tf.logical_not(tefaEm.stopCriterion.stopVar),
                               lambda: tefaEm.update(X=data),
                               lambda: tefaBcd.update(X=data))

            # update the global stop variable
            with tf.control_dependencies(updatedU):
                updatedStopVar = tf.assign(stopVar,
                                           tf.logical_and(tefaEm.stopCriterion.stopVar,
                                                          tefaBcd.stopCriterion.stopVar))

            with tf.control_dependencies([updatedStopVar]):
                step = tf.train.get_or_create_global_step()
                trainOp = tf.assign(step, step + 1)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=trainOp)

        est = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=path,
                                     warm_start_from=ws)

        return(est)
