import jax.numpy as np


def safe_log(x, eps=1e-10):
    """A logarithmic transform that is numerically safe."""
    return np.log(x + eps)


def ndims(x):
    """
    Return the ndims of a tensor.
    """
    return len(x.shape)


def pad_graph(F, A, to_size: int):
    """Pad F and A matrices with zeros to fit graph size."""
    pad_size = to_size - len(F)
    F = np.pad(F, [(0, pad_size), (0, 0)])
    A = np.pad(A, [(0, pad_size), (0, pad_size)])
    return F, A
