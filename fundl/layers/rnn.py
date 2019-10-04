"""RNN module."""
import jax.numpy as np
from jax import lax
from fundl.activations import relu, tanh
from fundl.utils import ndims
from functools import partial


def gru_step(params, h_t: np.array, x_t: np.array):
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

from jax import lax
from functools import partial



def gru(params: dict, x: np.array):
    """
    Gated recurrent unit layer.

    Implements the equations as stated here:
    https://en.wikipedia.org/wiki/Gated_recurrent_unit
    """
    h_t = np.ones(params["W_z"].shape[1])
    step_func = partial(gru_step, params)
    _, outputs = lax.scan(step_func, init=h_t, xs=x)
    return outputs


def lstm(params, x):
    """
    LSTM layer implemented according to equations here:
    https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """
    # outputs = []
    h_t = np.zeros(params["W_i"].shape[0])
    c_t = np.zeros(params["W_i"].shape[0])

    step_func = partial(lstm_step, params)
    _, outputs = lax.scan(step_func, init=(h_t, c_t), xs=x)
    return outputs
    # for _, row in enumerate(x):
    #     h_t, c_t = lstm_step(params, (h_t, c_t), row)
    #     outputs.append(h_t)
    # return np.vstack(outputs)


def lstm_step(params, carry, x_t):
    """
    One step in the lstm.

    :param params: Dictionary of parameters.
    :param x: One row from the input data.
    :param h_t: Hidden state vector, the output from previous step.
    :param c_t: Cell state vector, the output from previous step.
    """
    # transpose x
    h_t, c_t = carry
    x_t = np.transpose(x_t)
    # concatenate the previous hidden state with new input
    h_t = np.concatenate([h_t, x_t])

    i_t = relu(np.dot(params["W_i"], h_t) + params["b_i"])
    ctilde_t = tanh(np.dot(params["W_c"], h_t) + params["b_c"])
    f_t = relu(np.dot(params["W_f"], h_t) + params["b_f"])
    c_t = np.multiply(f_t, ctilde_t) + np.multiply(i_t, ctilde_t)

    o_t = relu(np.dot(params["W_o"], h_t) + params["b_o"])
    h_t = np.multiply(o_t, tanh(c_t))

    return (h_t, c_t), h_t
