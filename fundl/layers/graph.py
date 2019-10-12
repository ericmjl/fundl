import jax.numpy as np
from jax.lax import batch_matmul

from fundl.activations import identity


def mpnn(params, A, F, nonlin=identity):
    """
    message passing neural network layer

    performs one round of message passing according to the adjacency-like
    matrices present in As, and then does one "dense" matrix multiplication
    on top of the message passing.

    :param params: A dictionary of parameters.
    :param A: A 3D-tensor of adjacency matrices.
        1st dimension is the sample/batch dimension;
        2nd and 3rd dimension must be equal.
    :param F: A 3D-tensor of feature matrices.
        1st dimension is the sample/batch dimension;
        2nd dimension is the node dimension;
        3rd dimension is the feature dimension.
    :returns: F, a 3D-tensor of transformed features.
        1st dimension is the sample/batch dimension;
        2nd dimension is the node dimension;
        3rd dimension is the feature dimension.
    """
    F = batch_matmul(A, F)  # shape will be n_samps x n_nodes x n_feats
    F = np.dot(F, params["w"]) + params["b"]
    return nonlin(F)


def gather(F):
    """
    graph gathering layer

    :param Fs: A 3D-tensor of node-level feature matrices.
        1st dimension is the sample/batch dimension;
        2nd dimension is the node dimension, over which summation takes place;
        3rd dimension is the feature dimension.
    :returns: A 2D-tensor of graph-level feature matrices.
    """
    return np.sum(F, axis=1)
