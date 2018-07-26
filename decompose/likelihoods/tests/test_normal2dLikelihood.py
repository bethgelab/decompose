from unittest.mock import MagicMock
import pytest
import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.likelihoods.normal2dLikelihood import Normal2dLikelihood
from decompose.tests.fixtures import device, dtype
from decompose.distributions.distribution import UpdateType


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



def test_residuals(device, dtype):
    npdtype = dtype.as_numpy_dtype
    M, K, tau = (20, 30), 3, 0.1
    U = (tf.constant(np.random.normal(size=(K, M[0])).astype(npdtype)),
         tf.constant(np.random.normal(size=(K, M[1])).astype(npdtype)))
    noise = np.random.normal(size=M).astype(npdtype)
    data = tf.matmul(tf.transpose(U[0]), U[1]) + tf.constant(noise)

    lh = Normal2dLikelihood(M=M, K=K, tau=tau, dtype=dtype)
    lh.init(data=data)

    r = lh.residuals(U, data)

    assert(r.dtype == dtype)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        npr = sess.run(r)

    assert(np.allclose(noise.flatten(), npr, atol=1e-5, rtol=1e-5))
    tf.reset_default_graph()


def test_loss(device, dtype):
    npdtype = dtype.as_numpy_dtype
    M, K, tau = (20, 30), 3, 0.1
    U = (tf.constant(np.random.normal(size=(K, M[0])).astype(npdtype)),
         tf.constant(np.random.normal(size=(K, M[1])).astype(npdtype)))
    noise = np.random.normal(size=M).astype(npdtype)
    data = tf.matmul(tf.transpose(U[0]), U[1]) + tf.constant(noise)

    lh = Normal2dLikelihood(M=M, K=K, tau=tau, dtype=dtype)
    lh.init(data=data)

    loss = lh.loss(U, data)

    assert(loss.dtype == dtype)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nploss = sess.run(loss)

    assert(np.allclose(np.sum(noise**2), nploss, atol=1e-5, rtol=1e-5))
    tf.reset_default_graph()


def test_llh(device, dtype):
    npdtype = dtype.as_numpy_dtype
    M, K, tau = (20, 30), 3, 0.1
    U = (tf.constant(np.random.normal(size=(K, M[0])).astype(npdtype)),
         tf.constant(np.random.normal(size=(K, M[1])).astype(npdtype)))
    noise = np.random.normal(size=M).astype(npdtype)
    data = tf.matmul(tf.transpose(U[0]), U[1]) + tf.constant(noise)

    lh = Normal2dLikelihood(M=M, K=K, tau=tau, dtype=dtype)
    lh.init(data=data)

    llh = lh.llh(U, data)

    assert(llh.dtype == dtype)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        npllh = sess.run(llh)

    llhgt = np.sum(sp.stats.norm(loc=0., scale=1./np.sqrt(tau)).logpdf(noise))
    assert(np.allclose(llhgt, npllh, atol=1e-5, rtol=1e-5))
    tf.reset_default_graph()


def test_prepVars(device, f, dtype):
    npdtype = dtype.as_numpy_dtype
    M, K, tau = (20, 30), 3, 0.1
    npU = (np.random.normal(size=(K, M[0])).astype(npdtype),
           np.random.normal(size=(K, M[1])).astype(npdtype))
    U = (tf.constant(npU[0]), tf.constant(npU[1]))
    npnoise = np.random.normal(size=M).astype(npdtype)
    npdata = np.dot(npU[0].T, npU[1]) + npnoise
    data = tf.constant(npdata, dtype=dtype)

    lh = Normal2dLikelihood(M=M, K=K, tau=tau, dtype=dtype)
    lh.init(data=data)

    A, B, alpha = lh.prepVars(f, U, data)

    assert(A.dtype == dtype)
    assert(B.dtype == dtype)
    assert(alpha.dtype.base_dtype == dtype)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        npA, npB, npalpha = sess.run([A, B, alpha])

    if f == 0:
        Agt = np.dot(npdata, npU[1].T)
        Bgt = np.dot(npU[1], npU[1].T)
        assert(np.allclose(Agt, npA, atol=1e-5, rtol=1e-5))
        assert(np.allclose(Bgt, npB, atol=1e-5, rtol=1e-5))
        assert(np.allclose(tau, npalpha, atol=1e-5, rtol=1e-5))
    if f == 1:
        Agt = np.dot(npdata.T, npU[0].T)
        Bgt = np.dot(npU[0], npU[0].T)
        assert(np.allclose(Agt, npA, atol=1e-5, rtol=1e-5))
        assert(np.allclose(Bgt, npB, atol=1e-5, rtol=1e-5))
        assert(np.allclose(tau, npalpha, atol=1e-5, rtol=1e-5))
    tf.reset_default_graph()


def test_update(device, f, updateType, dtype):
    npdtype = dtype.as_numpy_dtype
    M, K, tau = (20, 30), 3, 0.1
    npU = (np.random.normal(size=(K, M[0])).astype(npdtype),
           np.random.normal(size=(K, M[1])).astype(npdtype))
    U = (tf.constant(npU[0]), tf.constant(npU[1]))
    npnoise = np.random.normal(size=M).astype(npdtype)
    npdata = np.dot(npU[0].T, npU[1]) + npnoise
    data = tf.constant(npdata, dtype=dtype)

    lh = Normal2dLikelihood(M=M, K=K, tau=tau, updateType=updateType)
    lh.init(data=data)
    lh.noiseDistribution.update = MagicMock()
    residuals = tf.ones_like(data)
    lh.residuals = MagicMock(return_value=residuals)

    lh.update(U, data)

    if updateType == UpdateType.ALL:
        lh.residuals.assert_called_once()
        lh.noiseDistribution.update.assert_called_once()
    else:
        lh.residuals.assert_not_called()
        lh.noiseDistribution.update.assert_not_called()
    tf.reset_default_graph()
