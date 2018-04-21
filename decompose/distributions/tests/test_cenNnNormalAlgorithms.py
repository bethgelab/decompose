import pytest
import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.distributions.cenNnNormalAlgorithms import CenNnNormalAlgorithms


@pytest.mark.slow
def test_cenNnNormal_fit():
    """Test if the fitted parameters match the true parameters."""
    tau = np.array([0.5, 1., 2.])
    nSamples = 100000

    nParameters = tau.shape[0]
    norm = sp.stats.norm(scale=1./np.sqrt(tau))
    data = np.abs(norm.rvs(size=(nSamples, nParameters)))
    parameters = {"tau": tf.constant(np.ones(nParameters))}
    tfData = tf.constant(data)
    parameters = CenNnNormalAlgorithms.fit(parameters=parameters,
                                           data=tfData)

    with tf.Session() as sess:
        parameters = sess.run(parameters)

    tauHat = parameters["tau"]
    assert(tauHat.shape == tau.shape)
    assert(np.allclose(tauHat, tau, atol=1e-1))
