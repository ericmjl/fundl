"""RNN layer tests."""
from fundl.layers.rnn import gru, lstm
from fundl.weights import add_gru_params, add_lstm_params

from hypothesis import given, settings
from hypothesis.strategies import integers

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


@given(
    input_dim=integers(min_value=1, max_value=10),
    output_dim=integers(min_value=1, max_value=10),
    n_samples=integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_lstm(input_dim, output_dim, n_samples):
    """Test for lstm layer."""
    params = dict()
    params = add_lstm_params(params, "lstm", input_dim, output_dim)
    x = npr.normal(size=(n_samples, input_dim))
    y = npr.normal(size=(n_samples, output_dim))

    out = lstm(params["lstm"], x)

    assert out.shape == y.shape
    
    
