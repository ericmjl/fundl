"""
Conventions:

- Loss functions that begin with an underscore are meant to be used inside of another
  loss function that is differentiable w.r.t. parameters.

  The signature should be `func(y, y_hat, **kwargs)`.

- The rest of the loss functions should be of a signature:

      `func(params, model, x, y, **kwargs)
"""

import jax.numpy as np

# https://github.com/google/jax/issues/190#issuecomment-451782333
from jax.flatten_util import ravel_pytree

from .layers.normalizing_flow import K_planar_flows


def _cross_entropy_loss(y, y_hat):
    """
    Also corresponds to the log likelihood of the Bernoulli
    distribution.

    Intended to be used inside of another function that differentiates w.r.t.
    parameters.
    """
    xent = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
    return np.mean(xent)


def _mse_loss(y, y_hat):
    """
    Intended to be used inside of another function that differentiates w.r.t.
    parameters.
    """
    return np.mean(np.power(y - y_hat, 2))


def _mae_loss(y, y_hat):
    """
    Intended to be used inside of another function that differentiates w.r.t.
    parameters.
    """
    return np.mean(np.abs(y - y_hat))


def _gaussian_kl(z_mean, z_log_var, mean=True):
    """
    KL divergence between the parameterizations of z_mean and z_log_var,
    and a unit Gaussian.
    """
    kl = -0.5 * (z_log_var - np.exp(z_log_var) - np.power(z_mean, 2) + 1)
    return kl


def mseloss(params, model, x, y):
    y_hat = model(params, x)
    return _mse_loss(y, y_hat)


def ae_loss(params, model, x, y):
    y_hat = model(params, x)
    return -np.sum(_cross_entropy_loss(y, y_hat))


def vae_loss(params, model, encoder, x, y, kwargs):
    """
    Variational autoencoder loss.

    kwargs supported:
    :param bool l2: Whether or not to do l2 regularization.
    """
    # Make predictions
    y_hat = model(params, x)
    z_mean, z_log_var = encoder(params, x)

    # CE-loss
    ce_loss = np.sum(_cross_entropy_loss(y, y_hat))
    # KL-loss
    kl_loss = np.sum(_gaussian_kl(z_mean, z_log_var))

    l2_loss = 0
    l2 = kwargs.pop("l2")
    if l2:
        if not isinstance(l2, bool):
            raise TypeError("l2 should be a boolean")
        # L2-loss
        flat_params, unflattener = ravel_pytree(params)
        l2_loss = np.dot(flat_params, flat_params)

    return -ce_loss + kl_loss + l2_loss


def planarflow_vae_loss(params, model, encoder, sampler, x, y, K, l2=True):
    """
    Loss function for normalizing flow VAEs.

    Assumes the sampling layer is Gaussian-distributed, and that the
    normalizing flows that are used are planar flows.

    :param params: Parameters in a dictionary.
    :param model: Function that specifies the model.
        Should have signature: (params, x, K)
    :param encoder: Function that specifies the encoder portion of the model.
        Should have signature: (params, x)
    :param encoder: Function that specifies the sampler portion of the model.
        Should have signature: (z_mean, z_log_var)
    :param x, y: inputs.
    :param K: A list of planar flow layer names.
    :param l2: Whether or not to use L2 regularization.
    """
    # Make predictions
    y_hat = model(params, x, K)
    z_mean, z_log_var = encoder(params, x)

    # CE-loss
    ce_loss = np.sum(_cross_entropy_loss(y, y_hat))
    # KL-loss
    kl_loss = np.sum(_gaussian_kl(z_mean, z_log_var))

    # L2-loss
    l2_loss = 0
    if l2:
        flat_params, unflattener = ravel_pytree(params)
        l2_loss = np.dot(flat_params, flat_params)

    z = sampler(z_mean, z_log_var)
    z, log_jacobians = K_planar_flows(params, z, K)
    # Sum of log-det-jacobians
    ldj_loss = np.sum(log_jacobians)

    return -ce_loss + kl_loss + l2_loss + ldj_loss
