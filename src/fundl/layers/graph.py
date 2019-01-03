import jax.numpy as np
from ..nonlinearities import identity


def mpnn(params, As, Fs, nonlin=identity):
    """
    message passing neural network layer

    performs one round of message passing according to the adjacency-like
    matrices present in As, and then does one "dense" matrix multiplication
    on top of the message passing.

    :param params: A dictionary of parameters.
    :param As: A list of ndarrays, adjacency-like.
    :param Fs: A list of ndarrays: node features-like.
    """
    outputs = []
    for a, f in zip(As, Fs):
        f = np.dot(a, f)
        f = np.dot(f, params["w"]) + params["b"]
        f = nonlin(f)
        outputs.append(f)
    return outputs
