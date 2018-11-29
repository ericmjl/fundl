import autograd.numpy as np

def cross_entropy_loss(y, y_hat, mean=True):
    xent = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
    if mean:
        return -np.mean(xent, axis=-1)
    else:
        return -xent


def mse_loss(y, y_hat):
    return np.mean(np.power(y - y_hat, 2), axis=-1)

def kl_divergence(z_mean, z_log_var, mean=True):
    kl = 1 + z_log_var - np.power(z_mean, 2) - np.exp(z_log_var)
    kl *= -0.5

    if mean:
        return np.mean(kl, axis=-1)
    else:
        return kl