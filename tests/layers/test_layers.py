import jax.numpy as np
import numpy.random as npr
import pytest
from hypothesis import given
from hypothesis.strategies import integers

from fundl.layers import batch_norm, dense, dropout


@given(
    input_size=integers(min_value=1, max_value=10),
    output_size=integers(min_value=1, max_value=10),
    n_samples=integers(min_value=1, max_value=10),
)
def test_dense(input_size, output_size, n_samples):
    params = dict()
    params["w"] = npr.normal(size=(input_size, output_size))
    params["b"] = npr.normal(size=(output_size,))

    x = npr.normal(size=(n_samples, input_size))

    output = dense(params, x)
    assert output.shape == (n_samples, output_size)


@pytest.mark.parametrize("p,expected", [(0, False), (1, True)])
@given(
    input_size=integers(min_value=1, max_value=10),
    output_size=integers(min_value=1, max_value=10),
)
def test_dropout(input_size, output_size, p, expected):
    x = npr.normal(size=(input_size, output_size))

    output = dropout(p, x)
    assert output.all() == expected
