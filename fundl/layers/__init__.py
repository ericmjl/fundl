import jax.numpy as np
from ..nonlinearities import identity


def dense(params, x, nonlin=identity):
    """
    "dense" layers are just affine shifts + nonlinearities.

    affine shifts are represented by multiplication by weights and adding biases.

    Assumes that params is a dictionary with 'w' and 'b' as keys.

    nonlinearity defaults to identity, but any elementwise numpy function can be applied.
    """
    a = nonlin(np.dot(x, params["w"]) + params["b"])
    return a


def dropout(p, x):
    """
    "dropout" layers randomly sets columns of x to zero.
    """
    mask = np.random.binomial(n=1, p=p, size=x.shape[1])
    return x * mask


def batch_norm(p, x):
    pass
