import pytest
import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.distributions.normalAlgorithms import NormalAlgorithms
from decompose.tests.fixtures import device, dtype


@pytest.mark.slow
def test_normal_sample(device, dtype):
    """Test whether the mean and the variance of the samples are correct."""
    npdtype = dtype.as_numpy_dtype
    mu = np.array([-1, 0., 1.]).astype(npdtype)
    tau = np.array([0.5, 1., 2.]).astype(npdtype)
    nSamples = 1000000

    nParameters = mu.shape[0]
    parameters = {"mu": tf.constant(mu, dtype=dtype),
                  "tau": tf.constant(tau, dtype=dtype)}
    tfNSamples = tf.constant(nSamples)
    r = NormalAlgorithms.sample(parameters=parameters,
                                nSamples=tfNSamples)

    assert(r.dtype == dtype)

    with tf.Session() as sess:
        r = sess.run(r)

    assert(r.shape == (nSamples, nParameters))
    muHat = np.mean(r, axis=0)
    assert(np.allclose(muHat, mu, atol=1e-1))
    tauHat = 1./np.var(r, axis=0)
    assert(np.allclose(tauHat, tau, atol=1e-1))


def test_normal_mode(device, dtype):
    """Test if the mode is equal to `mu`."""
    npdtype = dtype.as_numpy_dtype
    mu = np.array([-1, 0., 1.]).astype(npdtype)
    tau = np.array([0.5, 1., 2.]).astype(npdtype)

    nParameters = mu.shape[0]
    parameters = {"mu": tf.constant(mu, dtype=dtype),
                  "tau": tf.constant(tau, dtype=dtype)}
    mode = NormalAlgorithms.mode(parameters=parameters)

    assert(mode.dtype == dtype)

    with tf.Session() as sess:
        mode = sess.run(mode)

    assert(mode.shape == (nParameters,))
    assert(np.all(mode == mu))


def test_normal_pdf(device, dtype):
    """Test if the pdf is the same as reported by scipy."""
    npdtype = dtype.as_numpy_dtype
    mu = np.array([-1, 0., 1.]).astype(npdtype)
    tau = np.array([0.5, 1., 2.]).astype(npdtype)
    nSamples = 1000

    nParameters = mu.shape[0]
    data = np.random.random((nSamples, nParameters)).astype(npdtype)
    parameters = {"mu": tf.constant(mu, dtype=dtype),
                  "tau": tf.constant(tau, dtype=dtype)}
    tfData = tf.constant(data)
    probs = NormalAlgorithms.pdf(parameters=parameters,
                                 data=tfData)

    assert(probs.dtype == dtype)

    with tf.Session() as sess:
        probs = sess.run(probs)

    assert(probs.shape == (nSamples, nParameters))
    spProbs = sp.stats.norm(loc=mu, scale=1./np.sqrt(tau)).pdf(data)
    assert(np.allclose(probs, spProbs))


def test_normal_llh(device, dtype):
    """Test if the llh is the same as reported by scipy."""
    npdtype = dtype.as_numpy_dtype
    mu = np.array([-1, 0., 1.]).astype(npdtype)
    tau = np.array([0.5, 1., 2.]).astype(npdtype)
    nSamples = 1000

    nParameters = mu.shape[0]
    data = np.random.random((nSamples, nParameters)).astype(npdtype)
    parameters = {"mu": tf.constant(mu, dtype=dtype),
                  "tau": tf.constant(tau, dtype=dtype)}
    tfData = tf.constant(data)
    llh = NormalAlgorithms.llh(parameters=parameters,
                               data=tfData)

    assert(llh.dtype == dtype)

    with tf.Session() as sess:
        llh = sess.run(llh)

    assert(llh.shape == (nSamples, nParameters))
    spLlh = sp.stats.norm(loc=mu, scale=1./np.sqrt(tau)).logpdf(data)
    assert(np.allclose(llh, spLlh))


@pytest.mark.slow
def test_normal_fit(device, dtype):
    """Test if the fitted parameters match the true parameters."""
    npdtype = dtype.as_numpy_dtype
    mu = np.array([-1, 0., 1.]).astype(npdtype)
    tau = np.array([0.5, 1., 2.]).astype(npdtype)
    nSamples = 100000

    nParameters = mu.shape[0]
    norm = sp.stats.norm(loc=mu, scale=1./np.sqrt(tau))
    data = norm.rvs(size=(nSamples, nParameters)).astype(npdtype)
    parameters = {"mu": tf.constant(np.ones(nParameters), dtype=dtype),
                  "tau": tf.constant(np.ones(nParameters), dtype=dtype)}
    tfData = tf.constant(data)
    parameters = NormalAlgorithms.fit(parameters=parameters,
                                      data=tfData)

    assert(parameters['mu'].dtype == dtype)
    assert(parameters['tau'].dtype == dtype)


    with tf.Session() as sess:
        parameters = sess.run(parameters)

    muHat = parameters["mu"]
    assert(muHat.shape == mu.shape)
    assert(np.allclose(muHat, mu, atol=1e-1))

    tauHat = parameters["tau"]
    assert(tauHat.shape == tau.shape)
    assert(np.allclose(tauHat, tau, atol=1e-1))
