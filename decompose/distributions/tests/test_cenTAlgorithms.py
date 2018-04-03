import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.distributions.cenTAlgorithms import CenTAlgorithms


def test_t_fit():
    """Test if the fitted parameters match the true parameters."""
    nu = np.array([0.5, 1., 1.5])
    Psi = np.array([0.5, 1., 2.])
    nParameters = nu.shape[0]
    nSamples = 1000000
    shape = (nSamples, nParameters)
    nIterations = 100

    # sample from the distribution using the true parameters
    data = sp.stats.t(df=nu, scale=np.sqrt(Psi)).rvs(shape)
    tfData = tf.constant(data)

    # sample from the distribution using the true parameters
    parameters = {"nu": tf.constant(np.ones(nParameters)),
                  "Psi": tf.constant(np.ones(nParameters)),
                  "tau": tf.constant(np.ones((nSamples, nParameters)))}
    variables = {key: tf.get_variable(key, initializer=value)
                 for key, value in parameters.items()}

    # estimate the parameters from the random sample
    parameterUpdate = CenTAlgorithms.fit(parameters=variables,
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
    nuHat = parameters["nu"]
    assert(nuHat.shape == nu.shape)
    assert(np.allclose(nuHat, nu, atol=1e-1))
    PsiHat = parameters["Psi"]
    assert(PsiHat.shape == Psi.shape)
    assert(np.allclose(PsiHat, Psi, atol=1e-1))
    tf.reset_default_graph()
