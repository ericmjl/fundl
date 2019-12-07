from functools import partial

import jax.numpy as np
from jax import lax

from fundl.activations import relu, tanh


def lstm(params: dict, x: np.ndarray) -> np.ndarray:
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


def lstm_step(params: np.ndarray, carry: tuple, x_t: np.ndarray) -> np.ndarray:
    """
    One step in the lstm.

    :param params: Dictionary of parameters.
    :param x_t: One row from the input data.
    :param carry: h_t and c_t from previous step.
        h_t is the hidden state vector,
        while c_t is the cell state vector.
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
