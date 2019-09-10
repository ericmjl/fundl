import jax.numpy as np


def safe_log(x, eps=1e-10):
    """A logarithmic transform that is numerically safe."""
    return np.log(x + eps)


def ndims(x):
    """
    Return the ndims of a tensor.
    """
    return len(x.shape)
