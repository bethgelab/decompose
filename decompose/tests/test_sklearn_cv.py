import pytest
import numpy as np
import tensorflow as tf
from decompose.distributions.cenNormal import CenNormal
from decompose.sklearn import DECOMPOSE
from decompose.data.lowRank import LowRank
from decompose.cv.cv import Block


tf.logging.set_verbosity(tf.logging.INFO)


@pytest.mark.system
@pytest.mark.slow
def test_sklearn_cv(tmpdir):
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
    K, M_train, M_test = 3, [500, 100], [200, 100]
    lrData = LowRank(rank=K, M_train=M_train, M_test=M_test)

    # instantiate a model
    priors, K, dtype = [CenNormal(), CenNormal()], K, np.float32
    model = DECOMPOSE(modelDirectory, priors=priors, n_components=K,
                      cv=Block(nFolds=(3, 3), foldNumber=3), dtype=dtype)

    # train the model
    U0 = model.fit_transform(lrData.training)

    # get mask marking the training set
    testMask = model.testMask

    # check whether variance explained is between 0.95 and 1.
    U1 = model.components_
    testIndexes = testMask.flatten()
    testResiduals = (np.dot(U0.T, U1) - lrData.training).flatten()[testIndexes]
    testData = lrData.training.flatten()[testIndexes]
    testVarExpl = 1. - np.var(testResiduals)/np.var(testData)
    print("testVarExpl", testVarExpl)
    assert(0.95 <= lrData.var_expl_training((U0, U1)) <= 1.)
