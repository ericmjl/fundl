import jax.numpy as np
import numpy.random as npr

from fundl.activations import identity


def dense(params, x, nonlin=identity):
    """
    "dense" layers are just affine shifts + activation functions.

    Affine shifts are represented by multiplication by weights and adding biases.

    Assumes that params is a dictionary with 'w' and 'b' as keys.

    Activation defaults to identity, but any elementwise numpy function can be applied.

    :param params: A dictionary of weights. Should have "w" and "b" as keywords.
    :param x: Input data.
    :param activation: A callable that applies an elementwise activation function on the output array.
    """
    a = nonlin(np.dot(x, params["w"]) + params["b"])
    return a


def dropout(p, x):
    """
    "dropout" layers randomly sets columns of x to zero.

    Dropout here is implemented using a binomial mask
    of shape (1, n_columns).
    It is then broadcasted across the entire input matrix `x`.

    :param p: Probability of dropout.
        Should be a scalar float between 0 and 1.
    :param x: Outputs from previous layer.
    """
    mask = npr.binomial(n=1, p=p, size=x.shape[1])
    return x * mask


def batch_norm(p, x):
    pass
