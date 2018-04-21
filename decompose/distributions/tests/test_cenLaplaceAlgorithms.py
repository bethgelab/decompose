import pytest
import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.distributions.cenLaplaceAlgorithms import CenLaplaceAlgorithms


@pytest.mark.slow
def test_cenLaplace_fit():
    """Test if the fitted parameters match the true parameters."""
    beta = np.array([0.5, 1., 2.])
    nSamples = 100000

    nParameters = beta.shape[0]
    norm = sp.stats.laplace(scale=beta)
    data = norm.rvs(size=(nSamples, nParameters))
    parameters = {"beta": tf.constant(np.ones(nParameters))}
    tfData = tf.constant(data)
    parameters = CenLaplaceAlgorithms.fit(parameters=parameters,
                                          data=tfData)

    with tf.Session() as sess:
        parameters = sess.run(parameters)

    betaHat = parameters["beta"]
    assert(betaHat.shape == beta.shape)
    assert(np.allclose(betaHat, beta, atol=1e-1))
