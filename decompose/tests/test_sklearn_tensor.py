import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from decompose.distributions.cenNormal import CenNormal
from decompose.sklearn import DECOMPOSE
from decompose.data.lowRank import LowRank


tf.logging.set_verbosity(tf.logging.INFO)


def test_sklearn_tensor(tmpdir):
    """Tests the sk-learn interface of the tensor factorisation estimator.

    The test creates a `DECOMPOSE` object and applies its `fit_transform`
    method to some low rank training data. The learned filter banks have
    to reconstruct the data very well. Then unseen test data is transformed
    into the learned basis. The test data has to be recoverd from the
    transformed representation.
    """
    # create temporary directory where the model and its checkpoints are stored
    modelDirectory = str(tmpdir.mkdir("model"))

    # create a synthetic low rank dataset
    K, M_train, M_test = 3, [500, 100, 50], [500, 100, 50]
    lrData = LowRank(rank=K, M_train=M_train, M_test=M_test)

    # instantiate a model
    priors, K, dtype = [CenNormal, CenNormal, CenNormal], K, np.float32
    model = DECOMPOSE(modelDirectory, priors=priors, n_components=K,
                      dtype=dtype)

    # train the model
    U0 = model.fit_transform(lrData.training)

    # check whether variance explained is between 0.95 and 1.
    U1, U2 = model.components_
    assert(0.95 <= lrData.var_expl_training((U0, U1, U2)) <= 1.)

    # transform test data
    transformModelDirectory = str(tmpdir.mkdir("transformModel"))
    U0test = model.transform(transformModelDirectory=transformModelDirectory,
                             X=lrData.test)
    assert(0.95 <= lrData.var_expl_test((U0test, U1, U2)) <= 1.)
