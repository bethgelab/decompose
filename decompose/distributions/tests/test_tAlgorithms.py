import pytest
import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.distributions.tAlgorithms import TAlgorithms

@pytest.mark.slow
def test_t_sample():
    """Test whether the parameters can be recovered from many samples."""
    mu = np.array([-1, 0., 1.])
    nu = np.array([0.5, 1., 1.5])
    Psi = np.array([0.5, 1., 2.])
    nParameters = mu.shape[0]
    nSamples = 1000000
    shape = (nSamples, nParameters)
    nIterations = 100

    # sample from the distribution using the true parameters
    trueParameters = {"mu": tf.constant(mu),
                      "nu": tf.constant(nu),
                      "Psi": tf.constant(Psi),
                      "tau": tf.constant(np.random.random(shape))}
    tfData = TAlgorithms.sample(parameters=trueParameters,
                                nSamples=nSamples)

    # random initialize parameter estimates
    parameters = {"mu": tf.constant(np.ones(nParameters)),
                  "nu": tf.constant(np.ones(nParameters)),
                  "Psi": tf.constant(np.ones(nParameters)),
                  "tau": tf.constant(np.ones((nSamples, nParameters)))}
    variables = {key: tf.get_variable(key, initializer=value)
                 for key, value in parameters.items()}

    # estimate the parameters from the random sample
    parameterUpdate = TAlgorithms.fit(parameters=variables,
                                      data=tfData)
    varUpdates = {}
    for key, var in variables.items():
        varUpdates[key] = tf.assign(var, parameterUpdate[key])
    with tf.Session() as sess:
        # initialize variables
        for key, var in variables.items():
            sess.run(var.initializer)
        # update the variables
        for i in range(nIterations):
            sess.run(varUpdates)
        # get estimated parameters
        parameters = sess.run(variables)

    # check the estimations
    muHat = parameters["mu"]
    assert(muHat.shape == mu.shape)
    assert(np.allclose(muHat, mu, atol=1e-1))
    nuHat = parameters["nu"]
    assert(nuHat.shape == nu.shape)
    assert(np.allclose(nuHat, nu, atol=1e-1))
    PsiHat = parameters["Psi"]
    assert(PsiHat.shape == Psi.shape)
    assert(np.allclose(PsiHat, Psi, atol=1e-1))
    tf.reset_default_graph()


def test_t_mode():
    """Test if the mode is equal to `mu`."""
    mu = np.array([-1, 0., 1.])
    nu = np.array([0.5, 1., 1.5])
    Psi = np.array([0.5, 1., 2.])

    nParameters = mu.shape[0]
    parameters = {"mu": tf.constant(mu),
                  "nu": tf.constant(nu),
                  "Psi": tf.constant(Psi)}
    mode = TAlgorithms.mode(parameters=parameters)

    with tf.Session() as sess:
        mode = sess.run(mode)

    assert(mode.shape == (nParameters,))
    assert(np.all(mode == mu))


def test_t_pdf():
    """Test if the pdf is the same as reported by scipy."""
    mu = np.array([-1, 0., 1.])
    nu = np.array([0.5, 1., 1.5])
    Psi = np.array([0.5, 1., 2.])
    nSamples = 1000

    nParameters = mu.shape[0]
    parameters = {"mu": tf.constant(mu),
                  "nu": tf.constant(nu),
                  "Psi": tf.constant(Psi)}

    data = np.random.random((nSamples, nParameters))
    tfData = tf.constant(data)
    probs = TAlgorithms.pdf(parameters=parameters,
                            data=tfData)

    with tf.Session() as sess:
        probs = sess.run(probs)

    assert(probs.shape == (nSamples, nParameters))
    spProbs = sp.stats.t(df=nu, loc=mu, scale=np.sqrt(Psi)).pdf(data)
    assert(np.allclose(probs, spProbs))


def test_t_llh():
    """Test if the llh is the same as reported by scipy."""
    mu = np.array([-1, 0., 1.])
    nu = np.array([0.5, 1., 1.5])
    Psi = np.array([0.5, 1., 2.])
    nSamples = 1000

    nParameters = mu.shape[0]
    parameters = {"mu": tf.constant(mu),
                  "nu": tf.constant(nu),
                  "Psi": tf.constant(Psi)}

    data = np.random.random((nSamples, nParameters))
    tfData = tf.constant(data)
    llh = TAlgorithms.llh(parameters=parameters,
                          data=tfData)

    with tf.Session() as sess:
        llh = sess.run(llh)

    assert(llh.shape == (nSamples, nParameters))
    spLlh = sp.stats.t(df=nu, loc=mu, scale=np.sqrt(Psi)).logpdf(data)
    assert(np.allclose(llh, spLlh))


@pytest.mark.slow
def test_t_fit():
    """Test if the fitted parameters match the true parameters."""
    mu = np.array([-1, 0., 1.])
    nu = np.array([0.5, 1., 1.5])
    Psi = np.array([0.5, 1., 2.])
    nParameters = mu.shape[0]
    nSamples = 1000000
    shape = (nSamples, nParameters)
    nIterations = 100

    # sample from the distribution using the true parameters
    data = sp.stats.t(df=nu, loc=mu, scale=np.sqrt(Psi)).rvs(shape)
    tfData = tf.constant(data)

    # sample from the distribution using the true parameters
    parameters = {"mu": tf.constant(np.ones(nParameters)),
                  "nu": tf.constant(np.ones(nParameters)),
                  "Psi": tf.constant(np.ones(nParameters)),
                  "tau": tf.constant(np.ones((nSamples, nParameters)))}
    variables = {key: tf.get_variable(key, initializer=value)
                 for key, value in parameters.items()}

    # estimate the parameters from the random sample
    parameterUpdate = TAlgorithms.fit(parameters=variables,
                                      data=tfData)
    varUpdates = {}
    for key, var in variables.items():
        varUpdates[key] = tf.assign(var, parameterUpdate[key])
    with tf.Session() as sess:
        # initialize variables
        for key, var in variables.items():
            sess.run(var.initializer)
        # update the variables
        for i in range(nIterations):
            sess.run(varUpdates)
        # get estimated parameters
        parameters = sess.run(variables)

    # check the estimations
    muHat = parameters["mu"]
    assert(muHat.shape == mu.shape)
    assert(np.allclose(muHat, mu, atol=1e-1))
    nuHat = parameters["nu"]
    assert(nuHat.shape == nu.shape)
    assert(np.allclose(nuHat, nu, atol=1e-1))
    PsiHat = parameters["Psi"]
    assert(PsiHat.shape == Psi.shape)
    assert(np.allclose(PsiHat, Psi, atol=1e-1))
    tf.reset_default_graph()
