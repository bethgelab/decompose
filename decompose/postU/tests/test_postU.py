from unittest.mock import MagicMock
import pytest
import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.likelihoods.normal2dLikelihood import Normal2dLikelihood
from decompose.tests.fixtures import device, dtype
from decompose.distributions.distribution import UpdateType, Properties
from decompose.distributions.uniform import Uniform
from decompose.postU.postU import PostU


@pytest.fixture(scope="module",
                params=[0, 1])
def f(request):
    f = request.param
    return(f)


@pytest.fixture(scope="module",
                params=[UpdateType.ALL, UpdateType.ONLYLATENTS])
def updateType(request):
    updateType = request.param
    return(updateType)


def test_update(device, f, dtype):
    npdtype = dtype.as_numpy_dtype
    M, K, tau, F = (20, 30), 3, 0.1, 2
    npU = (np.random.normal(size=(K, M[0])).astype(npdtype),
           np.random.normal(size=(K, M[1])).astype(npdtype))
    U = [tf.constant(npU[0]), tf.constant(npU[1])]
    npnoise = np.random.normal(size=M).astype(npdtype)
    npdata = np.dot(npU[0].T, npU[1]) + npnoise
    data = tf.constant(npdata, dtype=dtype)

    lh = Normal2dLikelihood(M=M, K=K, tau=tau, dtype=dtype)
    lh.init(data=data)

    properties = Properties(persistent=True,
                            dtype=dtype)
    prior = Uniform(dummy=tf.constant(np.random.random(K).astype(npdtype),
                                      dtype=dtype),
                    properties=properties)

    postUf = PostU(lh, prior, f)

    Ufupdated = postUf.update(U, data, transform=False)

    for g in range(F):
        assert(Ufupdated.dtype == dtype)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        npUfupdated = sess.run(Ufupdated)

    assert(not np.allclose(npU[f], npUfupdated))

    tf.reset_default_graph()
