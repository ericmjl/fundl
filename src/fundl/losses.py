import jax.numpy as np
from fundl.layers.normalizing_flow import K_planar_flows

def cross_entropy_loss(y, y_hat, mean=True):
    """
    Also corresponds to the log likelihood of the Bernoulli
    distribution.
    """
    xent = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
    return xent


def mse_loss(y, y_hat):
    return np.mean(np.power(y - y_hat, 2), axis=-1)


def mae_loss(y, y_hat):
    return np.mean(np.abs(y - y_hat), axis=-1)


def gaussian_kl(z_mean, z_log_var, mean=True):
    """
    KL divergence between the parameterizations of z_mean and z_log_var,
    and a unit Gaussian.
    """
    kl = -0.5 * (z_log_var - np.exp(z_log_var) - np.power(z_mean, 2) + 1)
    return kl


def ae_loss(flat_params, unflattener, model, x, y):
    params = unflattener(flat_params)
    y_hat = model(params, x)
    return -np.sum(cross_entropy_loss(y, y_hat))


def vae_loss(flat_params, unflattener, model, encoder, x, y, kwargs):
    """
    Variational autoencoder loss.

    kwargs supported:
    :param bool l2: Whether or not to do l2 regularization.
    """
    # Make predictions
    params = unflattener(flat_params)
    y_hat = model(params, x)
    z_mean, z_log_var = encoder(params, x)

    # CE-loss
    ce_loss = np.sum(cross_entropy_loss(y, y_hat))
    # KL-loss
    kl_loss = np.sum(gaussian_kl(z_mean, z_log_var))

    l2_loss = 0
    l2 = kwargs.pop('l2')
    if l2:
        if not isinstance(l2, bool):
            raise TypeError('l2 should be a boolean')
        # L2-loss
        l2_loss = np.dot(flat_params, flat_params)

    return -ce_loss + kl_loss + l2_loss


def planarflow_vae_loss(flat_params, unflattener, model, encoder, sampler, x, y, K, l2=True):
    """
    Loss function for normalizing flow VAEs.

    Assumes the sampling layer is Gaussian-distributed, and that the
    normalizing flows that are used are planar flows.

    :param flat_params, unflattener: Parameters, flattened, and corresponding
        unflattener
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
    params = unflattener(flat_params)
    y_hat = model(params, x, K)
    z_mean, z_log_var = encoder(params, x)

    # CE-loss
    ce_loss = np.sum(cross_entropy_loss(y, y_hat))
    # KL-loss
    kl_loss = np.sum(gaussian_kl(z_mean, z_log_var))

    # L2-loss
    l2_loss = 0
    if l2:
        l2_loss = np.dot(flat_params, flat_params)
    params = unflattener(flat_params)

    z = sampler(z_mean, z_log_var)
    z, log_jacobians = K_planar_flows(params, z, K)
    # Sum of log-det-jacobians
    ldj_loss = np.sum(log_jacobians)

    return -ce_loss + kl_loss + l2_loss + ldj_loss
