"""RNN layer tests."""
import numpy as np
import numpy.random as npr
from hypothesis import given, settings
from hypothesis.strategies import integers

from fundl.layers.rnn import gru, lstm, mlstm1900_step, mlstm1900
from fundl.weights import add_gru_params, add_lstm_params, add_mlstm1900_params
from fundl.utils import sliding_window


@given(
    input_dim=integers(min_value=1, max_value=10),
    output_dim=integers(min_value=1, max_value=10),
    n_samples=integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_gru(input_dim, output_dim, n_samples):
    """Test for GRU layer."""
    params = dict()
    params = add_gru_params(
        params, "gru", input_dim=input_dim, output_dim=output_dim
    )
    x = npr.normal(size=(n_samples, input_dim))
    y = npr.normal(size=(n_samples, output_dim))

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


def test_mlstm1900_step():
    """
    Given fake data of the correct input shapes,
    make sure that the output shapes are also correct.
    """
    params = dict()
    params["wmx"] = npr.normal(size=(10, 1900))
    params["wmh"] = npr.normal(size=(1900, 1900))
    params["wx"] = npr.normal(size=(10, 7600))
    params["wh"] = npr.normal(size=(1900, 7600))
    params["b"] = npr.normal(size=(7600,))

    x_t = npr.normal(size=(1, 10))
    h_t = npr.normal(size=(1, 1900))
    c_t = npr.normal(size=(1, 1900))

    carry = (h_t, c_t)

    (h_t, c_t), _ = mlstm1900_step(params, carry, x_t)
    assert h_t.shape == (1, 1900)
    assert c_t.shape == (1, 1900)


def test_mlstm1900():
    """
    Given one fake sequence of window size 10 and k (tbd) sliding windows,
    ensure that we get out _an_ output from mLSTM1900.
    """
    x_full_length = npr.randint(0, 20, size=(300,))
    x = sliding_window(x_full_length, size=10)
    # assert False, print(x.shape)

    params = dict()
    params = add_mlstm1900_params(
        params, name="mlstm1900", input_dim=10, output_dim=1900
    )

    out = mlstm1900(params["mlstm1900"], x)
    assert out.shape == (x.shape[0], 1900)
