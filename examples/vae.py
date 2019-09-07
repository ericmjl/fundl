"""
Variational autoencoder written entirely using fundl.
"""

from fundl.layers import dense
from fundl.nonlinearities import sigmoid, tanh
import jax.numpy as np
from jax.random import normal


# First, we define the encoder, sampler, and decoder.
def encoder(params, x):
    a = dense(params["enc1"], x, nonlin=tanh)
    a = dense(params["enc2"], a, nonlin=tanh)
    z_mean = dense(params["mean"], a)
    z_log_var = dense(params["logvar"], a)
    return z_mean, z_log_var


def sampler(z_mean, z_log_var):
    """
    Implementation of reparameterization trick. Per sample, generate
    a new number sampled from from mean=z_mean,
    with variance scaled by z_log_var.

    This only works for Gaussians.
    """
    return z_mean + np.exp(z_log_var / 2) * normal(size=z_mean.shape)


def decoder(params, x):
    a = dense(params["dec1"], x, nonlin=tanh)
    a = dense(params["dec2"], a, nonlin=tanh)
    output = dense(params["dec3"], a, nonlin=sigmoid)
    return output


# Next, we define the VAE model.
def vae_model(params, x):
    z_mean, z_log_var = encoder(params, x)
    latent = sampler(z_mean, z_log_var)
    output = decoder(params, latent)
    return output


# For the heck of it, we can also define a corresponding autoencoder
# model. This just uses z_mean, and does not attempt sampling.
def ae_model(params, x):
    z_mean, z_log_var = encoder(params, x)
    output = decoder(params, z_mean)
    return output


# We then define the VAE loss. This is the cross entropy loss + KL divergence loss.
def vae_loss(flat_params, unflattener, model, x, y):
    # Make predictions
    params = unflattener(flat_params)
    y_hat = model(params, x)

    # Define KL-divergence loss
    z_mean, z_log_var = encoder(params, x)

    # Loss is sum of cross-entropy loss and KL loss.
    return np.mean(
        cross_entropy_loss(y, y_hat) + kl_divergence(z_mean, z_log_var)
    )
