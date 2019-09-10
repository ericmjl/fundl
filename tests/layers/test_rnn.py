"""RNN layer tests."""
from fundl.layers.rnn import gru
from fundl.weights import add_gru_params

import numpy as np
import numpy.random as npr


def test_gru():
    """Test for GRU layer."""
    params = dict()
    params = add_gru_params(params, "gru", input_dim=8, output_dim=1)
    x = npr.normal(size=(10, 8))
    y = npr.normal(size=(10, 1))

    out = gru(params["gru"], x)

    assert out.shape == y.shape
