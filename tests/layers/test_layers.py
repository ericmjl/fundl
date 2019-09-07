from fundl.layers import dense, dropout, batch_norm
import numpy.random as npr
import jax.numpy as np
from hypothesis import given
from hypothesis.strategies import integers


@given(
    integers(min_value=1, max_value=10),
    integers(min_value=1, max_value=10),
    integers(min_value=1, max_value=10),
)
def test_dense(input_size, output_size, n_samples):
    params = dict()
    params["w"] = npr.normal(size=(input_size, output_size))
    params["b"] = npr.normal(size=(output_size,))

    x = npr.normal(size=(n_samples, input_size))

    output = dense(params, x)
    assert output.shape == (n_samples, output_size)
