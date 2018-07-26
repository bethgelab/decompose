import pytest
import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.distributions.cenNnFullyElasticNetAlgorithms import CenNnFullyElasticNetAlgorithms


@pytest.mark.slow
def test_cenNnFullyElasticNet_sample():
    """Test whether the parameters can be recovered from many samples."""
    b = np.array([0., 0., 0., 1., 1., 1., 2., 2., 2.])

    mu = np.array([-2., 1., 0., 1., 1., 1., 2., 2., 2.])
    tau = np.array([.5, 1., 2., 1., 1., 1., 2., 2., 2.])

    betaExponential = np.array([1., 1., 1., .5, 1., 2., 2., 2., 2.])

    alpha = np.array([1., 1., 1., 1., 1., 1., 1., 1.5, 2.])
    beta = np.array([1., 1., 1., 1., 1., 1., 1., 2., 3.])
    tauLomax = np.array([1., 1., 1., 1., 1., 1., 1., 2., 3.])

    nParameters = alpha.shape[0]
    nSamples = 1000000
    nIterations = 2

    # sample from the distribution using the true parameters
    trueParameters = {"b": tf.constant(b),
                      "mu": tf.constant(mu),
                      "tau": tf.constant(tau),
                      "betaExponential": tf.constant(betaExponential),
                      "alpha": tf.constant(alpha),
                      "beta": tf.constant(beta),
                      "tauLomax": tf.constant(tauLomax)}
    tfData = CenNnFullyElasticNetAlgorithms.sample(parameters=trueParameters,
                                                   nSamples=nSamples)
    tfData = tf.Print(tfData, [tfData], "tfData")
    # random initialize parameter estimates
    parameters = {"b": tf.constant(np.ones(nParameters)),
                  "mu": tf.constant(np.ones(nParameters)),
                  "tau": tf.constant(np.ones(nParameters)),
                  "betaExponential": tf.constant(np.ones(nParameters)),
                  "alpha": tf.constant(np.ones(nParameters)),
                  "beta": tf.constant(np.ones(nParameters)),
                  "tauLomax": tf.constant(np.ones((nSamples, nParameters)))}
    variables = {key: tf.get_variable(key, initializer=value)
                 for key, value in parameters.items()}

    # estimate the parameters from the random sample
    parameterUpdate = CenNnFullyElasticNetAlgorithms.fit(parameters=variables,
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
    print("b", parameters['b'])
    alphaHat = parameters["alpha"]
    assert(alphaHat.shape == alpha.shape)
    assert(np.allclose(alphaHat, alpha, atol=1e-1))
    betaHat = parameters["beta"]
    assert(betaHat.shape == beta.shape)
    assert(np.allclose(betaHat, beta, atol=1e-1))
    tf.reset_default_graph()
