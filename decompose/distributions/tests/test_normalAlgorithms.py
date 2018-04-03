import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.distributions.normalAlgorithms import NormalAlgorithms


def test_normal_sample():
    """Test whether the mean and the variance of the samples are correct."""
    mu = np.array([-1, 0., 1.])
    tau = np.array([0.5, 1., 2.])
    nSamples = 1000000

    nParameters = mu.shape[0]
    parameters = {"mu": tf.constant(mu),
                  "tau": tf.constant(tau)}
    tfNSamples = tf.constant(nSamples)
    r = NormalAlgorithms.sample(parameters=parameters,
                                nSamples=tfNSamples)

    with tf.Session() as sess:
        r = sess.run(r)

    assert(r.shape == (nSamples, nParameters))
    muHat = np.mean(r, axis=0)
    assert(np.allclose(muHat, mu, atol=1e-1))
    tauHat = 1./np.var(r, axis=0)
    assert(np.allclose(tauHat, tau, atol=1e-1))


def test_normal_mode():
    """Test if the mode is equal to `mu`."""
    mu = np.array([-1, 0., 1.])
    tau = np.array([0.5, 1., 2.])

    nParameters = mu.shape[0]
    parameters = {"mu": tf.constant(mu),
                  "tau": tf.constant(tau)}
    mode = NormalAlgorithms.mode(parameters=parameters)

    with tf.Session() as sess:
        mode = sess.run(mode)

    assert(mode.shape == (nParameters,))
    assert(np.all(mode == mu))


def test_normal_pdf():
    """Test if the pdf is the same as reported by scipy."""
    mu = np.array([-1, 0., 1.])
    tau = np.array([0.5, 1., 2.])
    nSamples = 1000

    nParameters = mu.shape[0]
    data = np.random.random((nSamples, nParameters))
    parameters = {"mu": tf.constant(mu),
                  "tau": tf.constant(tau)}
    tfData = tf.constant(data)
    probs = NormalAlgorithms.pdf(parameters=parameters,
                                 data=tfData)

    with tf.Session() as sess:
        probs = sess.run(probs)

    assert(probs.shape == (nSamples, nParameters))
    spProbs = sp.stats.norm(loc=mu, scale=1./np.sqrt(tau)).pdf(data)
    assert(np.allclose(probs, spProbs))


def test_normal_llh():
    """Test if the llh is the same as reported by scipy."""
    mu = np.array([-1, 0., 1.])
    tau = np.array([0.5, 1., 2.])
    nSamples = 1000

    nParameters = mu.shape[0]
    data = np.random.random((nSamples, nParameters))
    parameters = {"mu": tf.constant(mu),
                  "tau": tf.constant(tau)}
    tfData = tf.constant(data)
    llh = NormalAlgorithms.llh(parameters=parameters,
                               data=tfData)

    with tf.Session() as sess:
        llh = sess.run(llh)

    assert(llh.shape == (nSamples, nParameters))
    spLlh = sp.stats.norm(loc=mu, scale=1./np.sqrt(tau)).logpdf(data)
    assert(np.allclose(llh, spLlh))


def test_normal_fit():
    """Test if the fitted parameters match the true parameters."""
    mu = np.array([-1, 0., 1.])
    tau = np.array([0.5, 1., 2.])
    nSamples = 100000

    nParameters = mu.shape[0]
    norm = sp.stats.norm(loc=mu, scale=1./np.sqrt(tau))
    data = norm.rvs(size=(nSamples, nParameters))
    parameters = {"mu": tf.constant(np.ones(nParameters)),
                  "tau": tf.constant(np.ones(nParameters))}
    tfData = tf.constant(data)
    parameters = NormalAlgorithms.fit(parameters=parameters,
                                      data=tfData)

    with tf.Session() as sess:
        parameters = sess.run(parameters)

    muHat = parameters["mu"]
    assert(muHat.shape == mu.shape)
    assert(np.allclose(muHat, mu, atol=1e-1))

    tauHat = parameters["tau"]
    assert(tauHat.shape == tau.shape)
    assert(np.allclose(tauHat, tau, atol=1e-1))
