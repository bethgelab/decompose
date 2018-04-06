import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.distributions.cenDoubleLomaxAlgorithms import CenDoubleLomaxAlgorithms


def test_cenDoubleLomax_sample():
    """Test whether the parameters can be recovered from many samples."""
    alpha = np.array([1., 1.5, 2.])
    beta = np.array([1., 2., 3.])
    nParameters = alpha.shape[0]
    nSamples = 1000000
    shape = (nSamples, nParameters)
    nIterations = 100

    # sample from the distribution using the true parameters
    trueParameters = {"alpha": tf.constant(alpha),
                      "beta": tf.constant(beta),
                      "tau": tf.constant(np.random.random(shape))}
    tfData = CenDoubleLomaxAlgorithms.sample(parameters=trueParameters,
                                             nSamples=nSamples)

    # random initialize parameter estimates
    parameters = {"alpha": tf.constant(np.ones(nParameters)),
                  "beta": tf.constant(np.ones(nParameters)),
                  "tau": tf.constant(np.ones((nSamples, nParameters)))}
    variables = {key: tf.get_variable(key, initializer=value)
                 for key, value in parameters.items()}

    # estimate the parameters from the random sample
    parameterUpdate = CenDoubleLomaxAlgorithms.fit(parameters=variables,
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
    alphaHat = parameters["alpha"]
    assert(alphaHat.shape == alpha.shape)
    assert(np.allclose(alphaHat, alpha, atol=1e-1))
    betaHat = parameters["beta"]
    assert(betaHat.shape == beta.shape)
    assert(np.allclose(betaHat, beta, atol=1e-1))
    tf.reset_default_graph()


def test_cenDoubleLomax_mode():
    """Test if the mode is equal to `mu`."""
    alpha = np.array([1., 1.5, 2.])
    beta = np.array([1., 2., 3.])

    nParameters = alpha.shape[0]
    trueParameters = {"alpha": tf.constant(alpha),
                      "beta": tf.constant(beta)}
    mode = CenDoubleLomaxAlgorithms.mode(parameters=trueParameters)

    with tf.Session() as sess:
        mode = sess.run(mode)

    assert(mode.shape == (nParameters,))
    assert(np.all(mode == np.zeros_like(alpha)))


def test_cenDoubleLomax_pdf():
    """Test if the pdf is the same as reported by scipy."""
    alpha = np.array([1., 1.5, 2.])
    beta = np.array([1., 2., 3.])
    nSamples = 1000

    nParameters = alpha.shape[0]
    parameters = {"alpha": tf.constant(alpha),
                  "beta": tf.constant(beta)}

    # test positive values
    data = np.random.random((nSamples, nParameters))
    tfData = tf.constant(data)
    probs = CenDoubleLomaxAlgorithms.pdf(parameters=parameters,
                                         data=tfData)
    with tf.Session() as sess:
        probs = sess.run(probs)

    assert(probs.shape == (nSamples, nParameters))
    spProbs = sp.stats.lomax(c=alpha, scale=beta).pdf(data)/2.
    assert(np.allclose(probs, spProbs))

    # test negative values
    data = -np.random.random((nSamples, nParameters))
    tfData = tf.constant(data)
    probs = CenDoubleLomaxAlgorithms.pdf(parameters=parameters,
                                         data=tfData)

    with tf.Session() as sess:
        probs = sess.run(probs)

    assert(probs.shape == (nSamples, nParameters))
    spProbs = sp.stats.lomax(c=alpha, scale=beta).pdf(np.abs(data))/2.
    assert(np.allclose(probs, spProbs))


def test_cenDoubleLomax_llh():
    """Test if the llh is the same as reported by scipy."""
    alpha = np.array([1., 1.5, 2.])
    beta = np.array([1., 2., 3.])
    nSamples = 1000

    nParameters = alpha.shape[0]
    parameters = {"alpha": tf.constant(alpha),
                  "beta": tf.constant(beta)}

    # test positive values
    data = np.random.random((nSamples, nParameters))
    tfData = tf.constant(data)
    llh = CenDoubleLomaxAlgorithms.llh(parameters=parameters,
                                       data=tfData)

    with tf.Session() as sess:
        llh = sess.run(llh)

    assert(llh.shape == (nSamples, nParameters))
    spLlh = sp.stats.lomax(c=alpha, scale=beta).logpdf(data) - np.log(2.)
    assert(np.allclose(llh, spLlh))

    # test negative values
    data = -np.random.random((nSamples, nParameters))
    tfData = tf.constant(data)
    llh = CenDoubleLomaxAlgorithms.llh(parameters=parameters,
                                       data=tfData)

    with tf.Session() as sess:
        llh = sess.run(llh)

    assert(llh.shape == (nSamples, nParameters))
    spLlh = sp.stats.lomax(c=alpha, scale=beta).logpdf(-data) - np.log(2.)
    assert(np.allclose(llh, spLlh))


def test_cenDoubleLomax_fit():
    """Test if the fitted parameters match the true parameters."""
    alpha = np.array([1., 1.5, 2.])
    beta = np.array([1., 2., 3.])
    nParameters = alpha.shape[0]
    nSamples = 1000000
    shape = (nSamples, nParameters)
    nIterations = 100

    # sample from the distribution using the true parameters
    data = sp.stats.lomax(c=alpha, scale=beta).rvs(shape)
    r = np.random.random(data.shape) > 0.5
    data[r] = -data[r]
    tfData = tf.constant(data)

    # sample from the distribution using the true parameters
    parameters = {"alpha": tf.constant(np.ones(nParameters)),
                  "beta": tf.constant(np.ones(nParameters)),
                  "tau": tf.constant(np.ones((nSamples, nParameters)))}
    variables = {key: tf.get_variable(key, initializer=value)
                 for key, value in parameters.items()}

    # estimate the parameters from the random sample
    parameterUpdate = CenDoubleLomaxAlgorithms.fit(parameters=variables,
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
    alphaHat = parameters["alpha"]
    assert(alphaHat.shape == alpha.shape)
    assert(np.allclose(alphaHat, alpha, atol=1e-1))
    betaHat = parameters["beta"]
    assert(betaHat.shape == beta.shape)
    assert(np.allclose(betaHat, beta, atol=1e-1))
    tf.reset_default_graph()
