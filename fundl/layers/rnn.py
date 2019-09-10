"""RNN module."""
import jax.numpy as np

from fundl.activations import sigmoid
from fundl.utils import ndims

# def rnn(params: dict, x: np.array):
#     """
#     Vanilla RNN layer.
#     """
#     h_t = np.ones(params[""])


def gru_step(params, x: np.array, h_t: np.array):
    """
    One step in the GRU.

    :param params: Dictionary of parameters.
    :param x: One row from the input data.
    :param h_t: History vector, the output from previous step.
    """
    # Transform x into a row vector with an explicit sample dimension.
    if ndims(x) == 1:
        x = np.reshape(x, newshape=(1, -1))
    z_t = sigmoid(
        np.dot(x, params["W_z"]) + np.dot(x, params["U_z"]) + params["b_z"]
    )
    r_t = sigmoid(
        np.dot(x, params["W_r"]) + np.dot(params["U_r"], h_t) + params["b_r"]
    )
    h_t = z_t * h_t + (1 - z_t) * np.tanh(
        np.dot(x, params["W_h"])
        + np.dot((r_t * h_t), params["U_h"])
        + params["b_h"]
    )
    return h_t


def gru(params: dict, x: np.array):
    """
    Gated recurrent unit layer.

    Implements the equations as stated here:
    https://en.wikipedia.org/wiki/Gated_recurrent_unit
    """
    outputs = []
    h_t = np.ones(params["W_z"].shape[1])
    for _, row in enumerate(x):
        h_t = gru_step(params, row, h_t)
        outputs.append(h_t)
    return np.vstack(outputs)
