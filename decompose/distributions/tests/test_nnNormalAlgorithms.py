import pytest
import numpy as np
import scipy as sp
import scipy.stats
import tensorflow as tf

from decompose.distributions.nnNormalAlgorithms import NnNormalAlgorithms
from decompose.tests.fixtures import device, dtype


# @pytest.mark.slow
# def test_nnNormal_sample(dtype):
#     """Test whether the mean and the variance of the samples are correct."""
#     npdtype = dtype.as_numpy_dtype
#     mu = np.array([-1, 0., 1.]).astype(npdtype)
#     tau = np.array([0.5, 1., 2.]).astype(npdtype)
#     nSamples = 1000000

#     nParameters = mu.shape[0]
#     parameters = {"mu": tf.constant(mu, dtype=dtype),
#                   "tau": tf.constant(tau, dtype=dtype)}
#     tfNSamples = nSamples
#     r = NnNormalAlgorithms.sample(parameters=parameters,
#                                   nSamples=tfNSamples)

#     assert(r.dtype == dtype)

#     with tf.Session() as sess:
#         r = sess.run(r)

#     assert(r.shape == (nSamples, nParameters))

#     sigma = 1./np.sqrt(tau)
#     alpha = -mu/sigma
#     normal = sp.stats.norm()
#     meanGt = mu + sigma*normal.pdf(alpha)/normal.cdf(-alpha)
#     mean = np.mean(r, axis=0)
#     assert(np.allclose(meanGt, mean, atol=1e-1))

#     varGt = sigma**2*(1 + alpha*normal.pdf(alpha)/normal.cdf(-alpha)
#                       - (normal.pdf(alpha)/normal.cdf(-alpha))**2)
#     var = np.var(r, axis=0)
#     assert(np.allclose(varGt, var, atol=1e-1))


# def test_nnNormal_mode(dtype):
#     """Test if the mode is equal to `mu`."""
#     npdtype = dtype.as_numpy_dtype
#     mu = np.array([-1, 0., 1.]).astype(npdtype)
#     tau = np.array([0.5, 1., 2.]).astype(npdtype)

#     nParameters = mu.shape[0]
#     parameters = {"mu": tf.constant(mu, dtype=dtype),
#                   "tau": tf.constant(tau, dtype=dtype)}
#     mode = NnNormalAlgorithms.mode(parameters=parameters)

#     assert(mode.dtype == dtype)

#     with tf.Session() as sess:
#         mode = sess.run(mode)

#     modeGt = mu
#     modeGt[modeGt < 0.] = 0.

#     assert(mode.shape == (nParameters,))
#     assert(np.all(mode == modeGt))


# def test_nnNormal_pdf(dtype):
#     """Test if the pdf is the same as reported by scipy."""
#     npdtype = dtype.as_numpy_dtype
#     mu = np.array([-1, 0., 1.]).astype(npdtype)
#     tau = np.array([0.5, 1., 2.]).astype(npdtype)
#     nSamples = 1000

#     nParameters = mu.shape[0]
#     data = np.random.random((nSamples, nParameters)).astype(npdtype)
#     parameters = {"mu": tf.constant(mu, dtype=dtype),
#                   "tau": tf.constant(tau, dtype=dtype)}
#     tfData = tf.constant(data)
#     probs = NnNormalAlgorithms.pdf(parameters=parameters,
#                                    data=tfData)

#     assert(probs.dtype == dtype)

#     with tf.Session() as sess:
#         probs = sess.run(probs)

#     assert(probs.shape == (nSamples, nParameters))
#     sigma = 1./np.sqrt(tau)
#     a, b = -mu/sigma, np.ones_like(mu)*np.inf
#     print(a.shape, b.shape, mu.shape, sigma.shape, data.shape)
#     for i in range(mu.shape[0]):
#         spProbsi = sp.stats.truncnorm.pdf(x=data[..., i], a=a[i], b=b[i],
#                                           loc=mu[i], scale=sigma[i])
#         assert(np.allclose(probs[..., i], spProbsi))


# def test_nnNormal_llh(dtype):
#     """Test if the llh is the same as reported by scipy."""
#     npdtype = dtype.as_numpy_dtype
#     mu = np.array([-1, 0., 1.]).astype(npdtype)
#     tau = np.array([0.5, 1., 2.]).astype(npdtype)
#     nSamples = 1000

#     nParameters = mu.shape[0]
#     data = np.random.random((nSamples, nParameters)).astype(npdtype)

#     parameters = {"mu": tf.constant(mu, dtype=dtype),
#                   "tau": tf.constant(tau, dtype=dtype)}
#     tfData = tf.constant(data)
#     llh = NnNormalAlgorithms.llh(parameters=parameters,
#                                  data=tfData)

#     assert(llh.dtype == dtype)

#     with tf.Session() as sess:
#         llh = sess.run(llh)

#     assert(llh.shape == (nSamples, nParameters))
#     sigma = 1./np.sqrt(tau)
#     a, b = -mu/sigma, np.ones_like(mu)*np.inf
#     print(a.shape, b.shape, mu.shape, sigma.shape, data.shape)
#     for i in range(mu.shape[0]):
#         spLlhi = sp.stats.truncnorm.logpdf(x=data[..., i], a=a[i], b=b[i],
#                                            loc=mu[i], scale=sigma[i])
#         assert(np.allclose(llh[..., i], spLlhi))


@pytest.mark.slow
def test_nnNormal_fit(dtype):
    """Test if the fitted parameters match the true parameters."""
    npdtype = dtype.as_numpy_dtype
    mu = np.array([-1, 0., 1.]).astype(npdtype)
    tau = np.array([0.5, 1., 2.]).astype(npdtype)
    sigma = 1./np.sqrt(tau)
    nSamples = 100000

    nParameters = mu.shape[0]
    a, b = -mu/sigma, np.ones_like(mu)*np.inf
    norm = sp.stats.truncnorm(a=a, b=b, loc=mu, scale=sigma)
    data = norm.rvs(size=(nSamples, nParameters)).astype(npdtype)
    parameters = {"mu": tf.constant(np.ones(nParameters), dtype=dtype),
                  "tau": tf.constant(np.ones(nParameters), dtype=dtype)}
    tfData = tf.constant(data)
    parameters = NnNormalAlgorithms.fit(parameters=parameters,
                                        data=tfData)

    assert(parameters['mu'].dtype == dtype)
    assert(parameters['tau'].dtype == dtype)

    with tf.Session() as sess:
        parameters = sess.run(parameters)

    muHat = parameters["mu"]
    assert(muHat.shape == mu.shape)
    assert(np.allclose(muHat, mu, atol=1e-1))

    tauHat = parameters["tau"]
    assert(tauHat.shape == tau.shape)
    assert(np.allclose(tauHat, tau, atol=1e-1))

    print(muHat)
    print(tauHat)
