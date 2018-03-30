import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.distributions.laplaceAlgorithms import LaplaceAlgorithms


def test_laplace_sample():
    """Test whether the mean and the variance of the samples are correct."""
    mu = np.array([-1, 0., 1.])
    beta = np.array([0.5, 1., 2.])
    nSamples = 1000000

    nParameters = mu.shape[0]
    parameters = {"mu": tf.constant(mu),
                  "beta": tf.constant(beta)}
    tfNSamples = tf.constant(nSamples)
    r = LaplaceAlgorithms.sample(parameters=parameters,
                                 nSamples=tfNSamples)

    with tf.Session() as sess:
        r = sess.run(r)

    assert(r.shape == (nSamples, nParameters))
    muHat = np.median(r, axis=0)
    assert(np.allclose(muHat, mu, atol=1e-1))
    betaHat = np.sqrt(np.var(r, axis=0)/2.)
    assert(np.allclose(betaHat, beta, atol=1e-1))


def test_laplace_mode():
    """Test if the mode is equal to `mu`."""
    mu = np.array([-1, 0., 1.])
    beta = np.array([0.5, 1., 2.])

    nParameters = mu.shape[0]
    parameters = {"mu": tf.constant(mu),
                  "beta": tf.constant(beta)}
    mode = LaplaceAlgorithms.mode(parameters=parameters)

    with tf.Session() as sess:
        mode = sess.run(mode)

    assert(mode.shape == (nParameters,))
    assert(np.all(mode == mu))


def test_laplace_pdf():
    """Test if the pdf is the same as reported by scipy."""
    mu = np.array([-1, 0., 1.])
    beta = np.array([0.5, 1., 2.])
    nSamples = 1000

    nParameters = mu.shape[0]
    data = np.random.random((nSamples, nParameters))
    parameters = {"mu": tf.constant(mu),
                  "beta": tf.constant(beta)}
    tfData = tf.constant(data)
    probs = LaplaceAlgorithms.pdf(parameters=parameters,
                                  data=tfData)

    with tf.Session() as sess:
        probs = sess.run(probs)

    assert(probs.shape == (nSamples, nParameters))
    spProbs = sp.stats.laplace(loc=mu, scale=beta).pdf(data)
    assert(np.allclose(probs, spProbs))


def test_laplace_llh():
    """Test if the llh is the same as reported by scipy."""
    mu = np.array([-1, 0., 1.])
    beta = np.array([0.5, 1., 2.])
    nSamples = 1000

    nParameters = mu.shape[0]
    data = np.random.random((nSamples, nParameters))
    parameters = {"mu": tf.constant(mu),
                  "beta": tf.constant(beta)}
    tfData = tf.constant(data)
    llh = LaplaceAlgorithms.llh(parameters=parameters,
                                data=tfData)

    with tf.Session() as sess:
        llh = sess.run(llh)

    assert(llh.shape == (nSamples, nParameters))
    spLlh = sp.stats.laplace(loc=mu, scale=beta).logpdf(data)
    assert(np.allclose(llh, spLlh))


def test_laplace_fit():
    """Test if the fitted parameters match the true parameters."""
    mu = np.array([-1, 0., 1.])
    beta = np.array([0.5, 1., 2.])
    nSamples = 100000

    nParameters = mu.shape[0]
    data = np.random.random((nSamples, nParameters))
    norm = sp.stats.laplace(loc=mu, scale=beta)
    data = norm.rvs(size=(nSamples, nParameters))
    parameters = {"mu": tf.constant(np.ones(nParameters)),
                  "beta": tf.constant(np.ones(nParameters))}
    tfData = tf.constant(data)
    parameters = LaplaceAlgorithms.fit(parameters=parameters,
                                       data=tfData)

    with tf.Session() as sess:
        parameters = sess.run(parameters)

    muHat = parameters["mu"]
    assert(muHat.shape == mu.shape)
    assert(np.allclose(muHat, mu, atol=1e-1))

    betaHat = parameters["beta"]
    assert(betaHat.shape == beta.shape)
    assert(np.allclose(betaHat, beta, atol=1e-1))
