import jax.numpy as np


def dense(params, x, nonlin=None):
    """
    "dense" layers are just affine shifts + nonlinearities.

    affine shifts are represented by multiplication by weights and adding biases.

    Assumes that params is a dictionary with 'w' and 'b' as keys.
    """
    a = np.dot(x, params["w"]) + params["b"]
    if nonlin:
        a = nonlin(a)
    return a


def dropout(p, x):
    """
    "dropout" layers randomly sets columns of x to zero.
    """
    mask = np.random.binomial(n=1, p=p, size=x.shape[1])
    return x * mask
