from functools import partial

import jax.numpy as np
from jax import lax

from fundl.activations import relu


def gru_step(params, h_t: np.array, x_t: np.array) -> np.ndarray:
    """
    One step in the GRU.

    :param params: Dictionary of parameters.
    :param x: One row from the input data.
    :param h_t: History vector, the output from previous step.
    """
    # Transform x into a row vector with an explicit sample dimension.
    # if ndims(x_t) == 1:
    #     x_t = np.reshape(x_t, newshape=(1, -1))
    z_t = relu(
        np.dot(x_t, params["W_z"]) + np.dot(x_t, params["U_z"]) + params["b_z"]
    )
    r_t = relu(
        np.dot(x_t, params["W_r"]) + np.dot(h_t, params["U_r"]) + params["b_r"]
    )
    h_t = z_t * h_t + (1 - z_t) * relu(
        np.dot(x_t, params["W_h"])
        + np.dot((r_t * h_t), params["U_h"])
        + params["b_h"]
    )
    return h_t, h_t


def gru(params: dict, x: np.ndarray) -> np.ndarray:
    """
    Gated recurrent unit layer.

    Implements the equations as stated here:
    https://en.wikipedia.org/wiki/Gated_recurrent_unit
    """
    h_t = np.ones(params["W_z"].shape[1])
    step_func = partial(gru_step, params)
    _, outputs = lax.scan(step_func, init=h_t, xs=x)
    return outputs
