import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from decompose.models.tensorFactorisation import TensorFactorisation
from decompose.stopCriterions.stopCriterion import StopHook
from decompose.distributions.cenNormal import CenNormal
from decompose.distributions.normal import Normal
from decompose.distributions.exponential import Exponential
from decompose.distributions.laplace import Laplace
from decompose.distributions.t import T
from decompose.distributions.lomax import Lomax
from decompose.distributions.cenNnNormal import CenNnNormal
from decompose.distributions.cenT import CenT
from decompose.distributions.cenNnT import CenNnT
from decompose.distributions.nnUniform import NnUniform
from decompose.distributions.cenDoubleLomax import CenDoubleLomax
from decompose.distributions.cenLaplace import CenLaplace
from decompose.data.random import Random
from decompose.stopCriterions.nIterations import NIterations


tf.logging.set_verbosity(tf.logging.INFO)


@pytest.fixture(params=[CenDoubleLomax, CenLaplace, CenNnNormal, CenNnT,
                        CenNormal, CenT, Exponential, Laplace, Lomax,
                        NnUniform, Normal, T])
def PriorDistribution(request):
    """A fixture that provides a prior distribution at a time."""
    prior = request.param
    return(prior)


@pytest.mark.system
@pytest.mark.slow
def test_all_priors(tmpdir, PriorDistribution):
    """Tests distributions in a tensor factorisation model.

    The test is useful to check for obious error such as shape
    mismatches or data type mismatches. The test does not check
    whether the priors are correctly implemented. The test fits a
    model with the specified priors to random data.
    """
    # create temporary directory where the model and its checkpoints are stored
    modelDirectory = str(tmpdir.mkdir("model"))

    # create a synthetic low rank dataset
    randomData = Random(M=(100, 200))

    # instantiate a model
    priors, K, dtype = [PriorDistribution(), PriorDistribution()], 3, np.float32
    tefa = TensorFactorisation.getEstimator(priors=priors,
                                            K=K,
                                            stopCriterionInit=NIterations(10),
                                            stopCriterionEM=NIterations(10),
                                            stopCriterionBCD=NIterations(10),
                                            dtype=tf.as_dtype(dtype),
                                            path=modelDirectory,
                                            device="/cpu:0")

    # create input_fna
    input_fn = randomData.input_fn

    # train the model
    tefa.train(input_fn=input_fn,
               hooks=[StopHook()])
