import pytest
import tensorflow as tf


@pytest.fixture(scope="module",
                params=["/cpu:0"])
def device(request):
    device = request.param
    with tf.device(device):
        yield device


@pytest.fixture(scope="module",
                params=[tf.float32, tf.float64])
def dtype(request):
    dtype = request.param
    return(dtype)
