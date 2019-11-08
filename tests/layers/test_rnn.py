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
    params = add_mlstm1900_params(params, "mlstm1900", 10, 1900)

    x_t = npr.normal(size=(1, 10))
    h_t = npr.normal(size=(1, 1900))
    c_t = npr.normal(size=(1, 1900))

    carry = (h_t, c_t)

    (h_t, c_t), _ = mlstm1900_step(params["mlstm1900"], carry, x_t)
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


def test_mlstm1900_example():
    # Set up an example
    sequence = np.array(
        [
            24,
            1,
            2,
            4,
            13,
            6,
            6,
            21,
            18,
            8,
            13,
            16,
            16,
            14,
            17,
            21,
            16,
            6,
            21,
            5,
            13,
            5,
            16,
            9,
            13,
            3,
            4,
            18,
            7,
            16,
            2,
            13,
            6,
            13,
            6,
            13,
            5,
            15,
            8,
            9,
            13,
            4,
            21,
            8,
            21,
            4,
            18,
            17,
            11,
            8,
            8,
            13,
            4,
            21,
            14,
            16,
            14,
            20,
            14,
            8,
            21,
            16,
            8,
            8,
            21,
            8,
            19,
            13,
            16,
            10,
            11,
            18,
            15,
            2,
            19,
            14,
            5,
            3,
            1,
            4,
            10,
            3,
            5,
            18,
            18,
            4,
            7,
            15,
            1,
            14,
            6,
            13,
            19,
            16,
            10,
            6,
            2,
            8,
            17,
            7,
            18,
            4,
            5,
            5,
            13,
            8,
            19,
            4,
            8,
            2,
            15,
            6,
            16,
            4,
            18,
            6,
            13,
            5,
            8,
            21,
            16,
            9,
            2,
            17,
            6,
            21,
            4,
            13,
            17,
            5,
            18,
            4,
            6,
            5,
            13,
            9,
            17,
            21,
            13,
            3,
            4,
            21,
            6,
            19,
            9,
            18,
            9,
            7,
            3,
            9,
            16,
            19,
            17,
            8,
            15,
            5,
            4,
            10,
            4,
            9,
            13,
            17,
            4,
            15,
            9,
            18,
            4,
            17,
            2,
            3,
            9,
            16,
            6,
            5,
            13,
            7,
            16,
            10,
            21,
            15,
            5,
            3,
            19,
            10,
            10,
            9,
            8,
            14,
            17,
            13,
            5,
            13,
            14,
            16,
            21,
            21,
            14,
            5,
            9,
            3,
            19,
            21,
            7,
            8,
            10,
            7,
            16,
            21,
            7,
            4,
            5,
            14,
            9,
            6,
            4,
            2,
            5,
            3,
            1,
            16,
            21,
            21,
            6,
            18,
            16,
            8,
            15,
            15,
            13,
            17,
            8,
            3,
            13,
            1,
            5,
            6,
            21,
            19,
            4,
        ]
    )

    x = sliding_window(sequence, size=10)
    params = dict()
    params = add_mlstm1900_params()
    # Pass through mLSTM1900

    # Check outputs
