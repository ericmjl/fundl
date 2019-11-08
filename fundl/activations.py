"""
Activations functions for fundl.
"""
import jax.numpy as np


def sigmoid(x):
    return 0.5 * np.tanh(x) + 0.5


def relu(x, alpha=0.1):
    return x * (x > 0) + alpha * x * (x <= 0)


def tanh(x):
    return np.tanh(x)


def leaky_relu(x, a):
    """Leaky ReLu"""
    pass


def identity(x):
    return x
