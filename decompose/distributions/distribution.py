from abc import ABCMeta, abstractmethod
from tensorflow.python.framework.dtypes import DType
from enum import Enum
from typing import Tuple, Dict, Type
import tensorflow as tf
from tensorflow import Tensor, Variable
import numpy as np

from decompose.distributions.productDistLookup import ProductDict
from decompose.distributions.algorithms import Algorithms


class DrawType(Enum):
    """Type indicating the behavior of a `Distribution` 's `draw` method."""
    SAMPLE = 1
    """Indicates that the `draw` method returns samples from the
    distribution by calling the `sample` method of the `Distribution`."""

    MODE = 2
    """Indicates that the `draw` method returns the
    mode of the distribution by calling the `mode` method of the
    `Distribution`."""


class UpdateType(Enum):
    """Type indicating the behavior of a distribution's `update` method."""
    ALL = 1
    """Indicates that the `update` method updates all variables."""

    ONLYLATENTS = 2
    """Indicates that the `update` method updates only the latent variables."""


class Distribution(metaclass=ABCMeta):
    """Common interface used for all distributions.

    This interface is different in several ways from tensorflow's
    distribution interface defined in `tf.distribution.Distribution`:
    Main differences are that this interface

    1) provides methods to fit the parameters of the distribution.
    2) allows distributions to be persistent i. e.
       the parameters are stored in `Variables` or not persistent
       i. e. the parameters are just `Tensor`s.
    3) This interface implements multiplications of distributions which
       are performed on the densities of the distributions.

    Arguments:
        name: `str` the name of the instance of a distribution.
        shape: `Tuple[int, ...]` the shape of the distribution.
        latentShape: `Tuple[int, ...]` the shape of latent variables.
        drawType: `DrawType` sets the behavior of the `draw` method.
        updateType: `UpdateType` sets the behavior of the `update` method.
        persistent: `bool` whether or not to store parameters in variables.
    """
    def __init__(self,
                 name: str,
                 shape: Tuple[int, ...],
                 latentShape: Tuple[int, ...],
                 algorithms: Type[Algorithms],
                 dtype: type = tf.float64,
                 drawType: DrawType = DrawType.SAMPLE,
                 updateType: UpdateType = UpdateType.ALL,
                 persistent: bool = True) -> None:

        self.__name = name
        self.__shape = shape
        self.__latentShape = latentShape
        self.__dtype = dtype
        self.__drawType = drawType
        self.__updateType = updateType
        self.__persistent = persistent
        self.algorithms = algorithms

    def _init(self, parameters: Dict[str, Tensor]) -> None:
        """Initializes the parameters and variables of the distribution.

        How the parameters are initizliaed depends on whether or not
        the distribution is `persistent`.  If it is not `persistent` the
        parameters are set to their values as specified in the
        `params` argument.  If the distribution is `persistent` the values
        in the `params` argument are only used as initial values for
        the variables and the parameters are set to the current state
        of the variables.

        This function has to be called by the constructor of a
        distribution.

        Arguments:
            params: `Dict[str, Tensor]` the (initial) values of the parameters.
        """
        self.__vars = {}  # type: Dict[str, tf.Variable]
        instanceName = self.name
        distributionName = type(self).__name__
        scope = f"{instanceName}/{distributionName}"
        with tf.variable_scope(scope):
            for parameterName, value in parameters.items():
                if self.persistent:
                    value = tf.get_variable(parameterName, dtype=self.dtype,
                                            initializer=value)
                    self.__vars[parameterName] = value
                setattr(self, parameterName, value)

    def get_parameters(self) -> Dict[str, Tensor]:
        parameters = {}
        for parameterName in self.parameterNames:
            parameters[parameterName] = getattr(self, parameterName)
        return(parameters)

    def set_parameters(self, parameters: Dict[str, Tensor]) -> None:
        for parameterName in parameters.keys():
            if parameterName not in self.parameterNames:
                raise ValueError
            setattr(self, parameterName, parameters[parameterName])

    @classmethod
    def random(cls,
               name: str,
               shape: Tuple[int, ...],
               latentShape: Tuple[int, ...],
               dtype: type = tf.float64,
               homogenous: bool = False,
               drawType: DrawType = DrawType.SAMPLE,
               updateType: UpdateType = UpdateType.ALL,
               persistent: bool = True) -> "Distribution":
        """Static method to create randomly initialized distributions.

        Arguments:
            name: `str` the name of the instance of a distribution.
            shape: `Tuple[int, ...]` the shape of the distribution.
            latentShape: `Tuple[int, ...]` the shape of latent variables.
            drawType: `DrawType` sets the behavior of the `draw` method.
            updateType: `UpdateType` sets the behavior of the `update` method.
            persistent: `bool` whether or not to store parameters in variables.

        Returns:
            `Distribution` object with randomly initialized parameters.
        """
        params = {
            "name": name,
            "drawType": drawType,
            "updateType": updateType,
            "persistent": persistent}

        initializers = cls.initializers(shape=shape, latentShape=latentShape,
                                        dtype=dtype)
        for parameter, initTensor in initializers.items():
            params[parameter] = initTensor
        distInstance = cls(**params)
        return(distInstance)

    def __mul__(self, other: object) -> "Distribution":
        """Multiplies the densities of two distibutions.

        Perform a multiplication of two probability densities.  The
        result of the multiplication is normalized and serves a
        density for a new persistent `Distribution` .

        Arguments:
            other: `object` providing the other density for the multiplication.

        Returns:
            Persistent `Distribution` with the resulting product as density.

        Raises:
            `ValueError` if the other operand is not of type `Distribution`.
        """
        if isinstance(other, Distribution):
            p = ProductDict().lookup(self, other)
            return(p)
        else:
            raise ValueError("Both factors must be of type Distribution")

    def __rmul__(self, other: object):
        """Multiplies the densities of two distibutions.

        Perform a multiplication of two probability densities.  The
        result of the multiplication is normalized and serves a
        density for a new persistent `Distribution` .

        Arguments:
            other: `object` providing the other density for the multiplication.

        Returns:
            Persistent `Distribution` with the resulting product as density.

        Raises:
            `ValueError` if the other operand is not of type `Distribution`.
        """
        return(self.__mul__(other))

    def __getitem__(self, key: object) -> "Distribution":
        """Indexing and slicing of `Distributions`s similar to numpy arrays.

        Slices and indexes all the parameters by applying the `key` parameter
        to each of its parameters `__getitem__` method. Then using the sliced
        and indexed parameters to create a new `Distribution` of the same type.
        The only difference is that such distributions are always not
        persistent.

        Arguments:
            key: `object` must be valid argument for slicing or indexing.

        Returns: `Distribution` of same type but only a subset of the original
            elements.
        """
        params = {
            "name": self.name,
            "drawType": self.drawType,
            "updateType": self.updateType,
            "persistent": False}
        for parameterName in self.parameterNames:
            params[parameterName] = getattr(self, parameterName)[key]
        i = type(self)(**params)
        return(i)

    @staticmethod
    @abstractmethod
    def initializers(shape: Tuple[int, ...] = (1,),
                     latentShape: Tuple[int, ...] = (),
                     dtype: DType = tf.float32) -> Dict[str, Tensor]:
        """Initializers of the parameters of the distribution.

        Draw random initialization values for each parameter matching the
        provided `shape`, `lantentShape`, and `dtype`. This method has to
        be implemented by concrete distributions to provide reasonable
        random initalizations used during `Distribution.random`.

        Arguments:
            shape: `Tuple[int, ...]` the shape of the distribution.
            latentShape: `Tuple[int, ...]` the latent shape of the
                distribution.
            dtype: `DType` the data type of the distribution.

        Returns:
            `Dict[str, Tensor]` map from parameter names to `Tensor` s
            containing the random initial values for the parameters.
        """
        ...

    def vars(self, parameterName: str) -> Variable:
        """Map a parameter name to its associated variable object.

        Returns the variable object for a given parameter name. The benefit
        over using `tf.get_variable` is that we do not need to specify
        the scope.

        Arguments:
            parameterName: `str` name of a parameter to look up the variable.

        Returns:
            `Variable` object associated with the `parameterName`.

        Raises:
            `ValueError` if the distribution is not persistent.
        """
        if not self.persistent:
            raise ValueError("Distribution must be persistent.")
        return(self.__vars[parameterName])

    @property
    def persistent(self) -> bool:
        """`bool` whether or not the distribution is persistent.

        A persistent distribution stores the parameter values in
        `Variable` s. A non persistent distribution does only
        hold the parameters in `Tensor` s.
        """
        return(self.__persistent)

    @property
    def parameterNames(self) -> Tuple[str, ...]:
        """`Tuple[str, ...]` parameter names of the distribution."""
        return(tuple(self.initializers().keys()))

    @property
    def shape(self) -> Tuple[int, ...]:
        """`Tuple[int, ...]` the shape of the distribution."""
        return(tuple(self.__shape.as_list()))

    @property
    def latentShape(self) -> Tuple[int, ...]:
        """`Tuple[int, ...]` the latent shape of the distribution."""
        return(self.__latentShape)

    @property
    def dtype(self) -> DType:
        """The dtype of the distribution."""
        return(self.__dtype)

    @property
    def name(self) -> str:
        """The name of the distribution"""
        return(self.__name)

    @property
    @classmethod
    @abstractmethod
    def nonNegative(cls) -> bool:
        """`bool` indicating whether the distribution is non-negative."""
        ...

    @property
    def drawType(self) -> DrawType:
        """`DrawType` indicating whether `draw` returns a sampel or the mode.

        If `drawType` is `DrawType.SAMPLE` then `draw` returns a sample from
        the distribution. If `drawType` is `DrawType.MODE` then `draw` returns
        the mode of the distribution.
        """
        return(self.__drawType)

    @property
    def updateType(self) -> UpdateType:
        """`UpdateType` indicating whether `update` updates all parameters.

        If `updateType` is `UpdateType.ALL` then `update` will update all
        parameters. If `updateType` is `UpdateType.ONLYLATENTS` then `update`
        will only update the latents.
        """
        return(self.__updateType)

    @drawType.setter
    def drawType(self, drawType: DrawType):
        self.__drawType = drawType

    @updateType.setter
    def updateType(self, updateType: UpdateType):
        self.__updateType = updateType

    @property
    @abstractmethod
    def homogenous(self) -> bool:
        """`bool` indicating parameter sharing throughout the distribution.

        A homogenous distribution uses the same set of parameters for
        all its elements. Estimating the parameters of a homogenous
        distribution will flatten the whole data and then fit a singe set of
        parameters using the flattened data. The fitted parameter set is
        assigned homogenously to all its elements.
        """
        ...

    def draw(self) -> Tensor:
        """Draw a realization from the distribution i. e. a sample or the mode.

        If `drawType` is equal to `DrawType.SAMPLE` the function returns a
        sample. If `drawType` is equal to `DrawType.MODE` the function returns
        the mode of the distribution.

        Returns:
            `Tensor` of the same shape as the distribution containing either
            a sample from the distribution or the mode of the distribution.
        """
        if self.drawType == DrawType.SAMPLE:
            return(self.sample(1))[..., 0]
        else:
            return(self.mode())

    def update(self, data: Tensor) -> None:
        """Update the parameters of the distribution given the data.

        If `updateType` is `UpdateType.ALL` then all parameters
        will be updated by calling `fit`. If `updateType` is
        `UpdateType.ONLYLATENTS` then only the latents will be updated
        by calling `fitLatents`.

        Arguments:
            data: `Tensor` data used to fit the parameters.
        """
        if self.updateType == UpdateType.ALL:
            self.fit(data)
        else:
            self.fitLatents(data)

    @abstractmethod
    def cond(self) -> "Distribution":
        """Conditions the distribution on its latent variables.

        The distribution is conditioned on the current state of its
        latent variables. The result is a non persistent `Distribution`
        where the shape is the concatenation of `shape` and `latentShape`
        of the original distribution.

        Returns:
            The conditional `Distribution` of itself
            when it is condition on its latent variables.
        """
        ...

    def sample(self, nSamples: int) -> Tensor:
        """Draw samples from the distribution.

        Draw `nSamples` from the distribution. The result is a `Tensor`
        where the shape is the same as the shape of the distribution but
        with one additional dimension added to the end representing the
        samples.

        Arguments:
            nSamples: `int` specifying the number of samples to draw.

        Returns:
            `Tensor` containing the drawn samples.
        """
        samples = self.algorithms.sample(parameters=self.get_parameters(),
                                         nSamples=nSamples)
        return(samples)

    def mode(self) -> Tensor:
        """The mode of the distribution.

        Calculates the mode for each element of the distribution.

        Returns:
            `Tensor` of the same shape as the distribution containing the
            modes.
        """
        mode = self.algorithms.mode(parameters=self.get_parameters())
        return(mode)

    def fit(self, data: Tensor) -> None:
        """Estimate the parameters of the distribution given `data`.

        Estimate the parameters of the distribution using maximum
        likelihood given the `data`. The shape of `data` is expected to
        be the same as the shape of the distribution but with one
        additional dimension added to the end representing the samples.

        Arguments:
            data: `Tensor` used to fit the parameters.
        """
        parameters = self.get_parameters()
        updatedParameters = self.algorithms.fit(parameters=parameters,
                                                data=data)
        self.set_parameters(updatedParameters)

    def fitLatents(self, data: Tensor) -> None:
        """Estimate the latent parameters of the distribution given `data`.

        Estimate the latent parameters of the distribution using maximum
        likelihood given the `data`. The shape of `data` is expected to
        be the same as the shape of the distribution but with one
        additional dimension added to the end representing the samples.

        Arguments:
            data: `Tensor` used to fit the latent parameters.
        """
        parameters = self.get_parameters()
        updatedParameters = self.algorithms.fitLatents(parameters=parameters,
                                                       data=data)
        self.set_parameters(updatedParameters)

    def pdf(self, data: Tensor):
        """Evaluate the density function on the data `data`.

        The probability density is evaluated for each element of `data`
        such that the dimension of `data` must broadcast with the shape of
        the distribution.

        Arguments:
            data: `Tensor` data to evaluate the probablity density on.

        Returns:
            `Tensor` of the same size as `data` with the probablities.
        """
        pdf = self.algorithms.pdf(parameters=self.get_parameters(), data=data)
        return(pdf)

    def llh(self, data: Tensor) -> Tensor:
        """Evaluate the log likelihood on the data `data`.

        The log likelihood is evaluated for each element of `data` such that
        the dimension of `data` must broadcast with the shape of the
        distribution.

        Arguments:
            data: `Tensor` data to evaluate the log likelihood on.

        Returns:
            `Tensor` of the same size as `data` with the log likelihoods.
        """
        llh = self.algorithms.llh(parameters=self.get_parameters(), data=data)
        return(llh)

    @classmethod
    def getEstimator(cls,
                     distType: Type["Distribution"],
                     path: str = "/tmp") -> tf.estimator.Estimator:
        """Create a `Estimator` for the parameters of a distribution.

        Instantiates an `Estimator` that can estimate the maximum likelihood
        parameters of a distribution of the type `distType`.

        Arguments:
            distType: `Type[Distribution]` sets the distribution type.
            path: `str` directory where model parameters are stored or loaded.

        Returns:
            `tf.estimator.Estimator` instance that can estimate the parameters
            of a distribution of type `distType`.
        """

        def model_fn(features, labels, mode):
            labels = list(features.keys())
            assert len(labels) == 1
            data = features[labels[0]]
            dtype = data.dtype
            nVars, nSamples = tuple(data.get_shape().as_list())

            distribution = distType.random(name="est",
                                           shape=(nVars,),
                                           latentShape=(nSamples,),
                                           persistent=True,
                                           dtype=dtype)

            # PREDICT
            if mode == tf.estimator.ModeKeys.PREDICT:
                raise ValueError

            # EVAL
            if mode == tf.estimator.ModeKeys.EVAL:
                loss = tf.reduce_sum((data-distribution.mu[..., None])**2)
                return tf.estimator.EstimatorSpec(
                    mode,
                    loss=loss)

            # TRAIN
            if mode == tf.estimator.ModeKeys.TRAIN:
                loss = tf.reduce_sum((data-distribution.mu[..., None])**2)
                distribution.fit(data)
                with tf.control_dependencies([distribution.mu,
                                              distribution.tau]):
                    step = tf.train.get_or_create_global_step()
                    train_op_normal = tf.assign(step, step + 1)
                return tf.estimator.EstimatorSpec(
                    mode,
                    loss=loss,
                    train_op=train_op_normal)

            raise ValueError
        est = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=path)

        return(est)


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

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        if obj.persistent:
            value = tf.assign(obj.vars(self.name), value)
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
