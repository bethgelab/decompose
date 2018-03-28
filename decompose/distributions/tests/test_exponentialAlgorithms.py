import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.distributions.exponentialAlgorithms import ExponentialAlgorithms


def test_exponential_sample():
    """Test if the mean of the samples equals the scale `beta`."""
    beta = np.array([0.5, 1., 2.])
    nSamples = 1000000

    nBetas = beta.shape[0]
    parameters = {"beta": tf.constant(beta)}
    tfNSamples = tf.constant(nSamples)

    r = ExponentialAlgorithms.sample(parameters=parameters,
                                     nSamples=tfNSamples)

    with tf.Session() as sess:
        r = sess.run(r)

    assert(r.shape == (nSamples, nBetas))
    betaHat = np.mean(r, axis=0)
    assert(np.allclose(betaHat, beta, atol=1e-1))


def test_exponential_mode():
    """Test if the mode is 0 for all elements."""
    beta = np.array([0.5, 1., 2.])

    nBetas = beta.shape[0]
    parameters = {"beta": tf.constant(beta)}

    mode = ExponentialAlgorithms.mode(parameters=parameters)

    with tf.Session() as sess:
        mode = sess.run(mode)

    assert(mode.shape == (nBetas,))
    assert(np.all(mode == 0.))


def test_exponential_pdf():
    """Test if the pdf is the same as reported by scipy."""
    beta = np.array([0.5, 1., 2.])
    nSamples = 1000

    nBetas = beta.shape[0]
    data = np.random.random((nSamples, nBetas))

    parameters = {"beta": tf.constant(beta)}
    tfData = tf.constant(data)
    probs = ExponentialAlgorithms.pdf(parameters=parameters,
                                      data=tfData)

    with tf.Session() as sess:
        probs = sess.run(probs)

    assert(probs.shape == (nSamples, nBetas))
    spProbs = sp.stats.expon(scale=beta).pdf(data)
    assert(np.allclose(probs, spProbs))


def test_exponential_llh():
    """Test if the llh is the same as reported by scipy."""
    beta = np.array([0.5, 1., 2.])
    nSamples = 1000

    nBetas = beta.shape[0]
    data = np.random.random((nSamples, nBetas))

    parameters = {"beta": tf.constant(beta)}
    tfData = tf.constant(data)
    llh = ExponentialAlgorithms.llh(parameters=parameters,
                                    data=tfData)

    with tf.Session() as sess:
        llh = sess.run(llh)

    assert(llh.shape == (nSamples, nBetas))
    spLlh = sp.stats.expon(scale=beta).logpdf(data)
    assert(np.allclose(llh, spLlh))


def test_exponential_fit():
    """Test if the fitted parameters match the true parameters."""
    beta = np.array([0.5, 1., 2.])
    nSamples = 100000
    nBetas = beta.shape[0]
    data = np.random.random((nSamples, nBetas))
    data = sp.stats.expon(scale=beta).rvs(size=(nSamples, nBetas))

    parameters = {"beta": tf.constant(np.ones(nBetas))}
    tfData = tf.constant(data)
    parameters = ExponentialAlgorithms.fit(parameters=parameters,
                                           data=tfData)

    with tf.Session() as sess:
        parameters = sess.run(parameters)

    betaHat = parameters["beta"]

    assert(betaHat.shape == beta.shape)
    assert(np.allclose(betaHat, beta, atol=1e-1))
