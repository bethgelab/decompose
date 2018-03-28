import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.distributions.cenNormalAlgorithms import CenNormalAlgorithms


def test_cenNormal_fit():
    """Test if the fitted parameters match the true parameters."""
    tau = np.array([0.5, 1., 2.])
    nSamples = 100000

    nParameters = tau.shape[0]
    norm = sp.stats.norm(scale=1./np.sqrt(tau))
    data = norm.rvs(size=(nSamples, nParameters))
    parameters = {"tau": tf.constant(np.ones(nParameters))}
    tfData = tf.constant(data)
    parameters = CenNormalAlgorithms.fit(parameters=parameters,
                                         data=tfData)

    with tf.Session() as sess:
        parameters = sess.run(parameters)

    tauHat = parameters["tau"]
    assert(tauHat.shape == tau.shape)
    assert(np.allclose(tauHat, tau, atol=1e-1))
