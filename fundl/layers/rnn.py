"""RNN module."""
import logging
from functools import partial

import jax.numpy as np
from jax import lax

from fundl.activations import relu, sigmoid, tanh
from fundl.utils import l2_normalize, ndims

logging.basicConfig(filename="/var/fundl.rnn.log.txt", level=logging.INFO, filemode="w")


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
    z_t = relu(np.dot(x_t, params["W_z"]) + np.dot(x_t, params["U_z"]) + params["b_z"])
    r_t = relu(np.dot(x_t, params["W_r"]) + np.dot(h_t, params["U_r"]) + params["b_r"])
    h_t = z_t * h_t + (1 - z_t) * relu(
        np.dot(x_t, params["W_h"]) + np.dot((r_t * h_t), params["U_h"]) + params["b_h"]
    )
    return h_t, h_t


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


def lstm_step(params, carry, x_t):
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


def mlstm1900(params, x):
    """
    LSTM layer implemented according to UniRep,
    found here:
    https://github.com/churchlab/UniRep/blob/master/unirep.py#L43

    This layer processes one encoded sequence at a time,
    passed as a two dimensional array, with number of rows
    being number of sliding windows, and number of columns
    being the size of the sliding window (for the exact
    reimplementation, window size is fixed to length 10)

    :param params: All weights and biases for a single
        mlstm1900 rnn cell.
    :param x: One sequence, sliced by window size.
    """
    h_t = np.zeros(params["wmh"].shape[0])
    c_t = np.zeros(params["wmh"].shape[0])

    step_func = partial(mlstm1900_step, params)
    _, outputs = lax.scan(step_func, init=(h_t, c_t), xs=x)
    return outputs


def mlstm1900_step(params, carry, x_t):
    """
    Implementation of mLSTMCell from UniRep paper, with weight normalization.

    Exact source code reference:
    https://github.com/churchlab/UniRep/blob/master/unirep.py#L75

    Shapes of parameters:
    - wmx: 10, 1900
    - wmh: 1900, 1900
    - wx: 10, 7600
    - wh: 1900, 7600

    - gmx: 1900
    - gmh: 1900
    - gx: 7600
    - gh: 7600

    - b: 7600

    Shapes of inputs:
    - x_t: (1, 10)
    - carry:
        - h_t: (1, 1900)
        - c_t: (1, 1900)
    """
    h_t, c_t = carry

    # Perform weight normalization first
    # (Corresponds to Line 113).
    # In the original implementation, this is toggled by a boolean flag,
    # but here we are enabling it by default.
    params["wx"] = l2_normalize(params["wx"], axis=0) * params["gx"]
    params["wh"] = l2_normalize(params["wh"], axis=0) * params["gh"]
    params["wmx"] = l2_normalize(params["wmx"], axis=0) * params["gmx"]
    params["wmh"] = l2_normalize(params["wmh"], axis=0) * params["gmh"]

    # logging.debug(f"wx: {params['wx']}")
    # logging.debug(f"wh: {params['wh']}")
    # logging.debug(f"wmx: {params['wmx']}")
    # logging.debug(f"wmh: {params['wmh']}")

    # Shape annotation
    # (:, 10) @ (10, 1900) * (:, 1900) @ (1900, 1900) => (:, 1900)
    m = np.matmul(x_t, params["wmx"]) * np.matmul(h_t, params["wmh"])
    logging.debug(f"m: {m}")

    # (:, 10) @ (10, 7600) * (:, 1900) @ (1900, 7600) + (7600, ) => (:, 7600)
    z = np.matmul(x_t, params["wx"]) + np.matmul(m, params["wh"]) + params["b"]
    logging.debug(f"z: {z}")

    # Splitting along axis 1, four-ways, gets us (:, 1900) as the shape
    # for each of i, f, o and u
    i, f, o, u = np.split(z, 4, axis=-1)  # input, forget, output, update
    logging.debug(f"i: {i}")
    logging.debug(f"f: {f}")
    logging.debug(f"o: {o}")
    logging.debug(f"u: {u}")

    # Elementwise transforms here.
    # Shapes are are (:, 1900) for each of the four.
    i = sigmoid(i, version="exp")
    logging.debug(f"i: {i}")
    f = sigmoid(f, version="exp")
    logging.debug(f"f: {f}")
    o = sigmoid(o, version="exp")
    logging.debug(f"o: {o}")
    u = tanh(u)
    logging.debug(f"u: {u}")

    # (:, 1900) * (:, 1900) + (:, 1900) * (:, 1900) => (:, 1900)
    c_t = f * c_t + i * u
    logging.debug(f"c_t: {c_t}")

    # (:, 1900) * (:, 1900) => (:, 1900)
    h_t = o * tanh(c_t)
    logging.debug(f"h_t: {h_t}")

    # h, c each have shape (:, 1900)
    return (h_t, c_t), h_t  # returned this way to match rest of fundl API.
