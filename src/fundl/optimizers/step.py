import autograd.numpy as np


def adam_step(op: dict, g: np.ndarray, i, flat_params: np.ndarray, weight_decay=False):
    """
    One step of the adamW optimizer.

    :param op: optimizer parameters
    :param g: derivative of loss function w.r.t. params. Should be same shape
        as flat_params.
    :param i: the iteration number
    :param flat_params: Flattened parameter vector.
    :param weight_decay: Whether to use weight decay or not. Requires a key 'wd' in
        the optimizer parameters dictionary.
    """
    op["m"] = (1 - op["b1"]) * g + op["b1"] * op["m"]  # First  moment estimate.
    op["v"] = (1 - op["b2"]) * (g ** 2) + op["b2"] * op["v"]  # Second moment estimate.
    mhat = op["m"] / (1 - op["b1"] ** (i + 1))  # Bias correction.
    vhat = op["v"] / (1 - op["b2"] ** (i + 1))
    if weight_decay:
        wd = op.get("wd")
        flat_params = (
            flat_params
            - op["step_size"] * mhat / (np.sqrt(vhat) + op["eps"])
            - op["wd"] * flat_params
        )
    else:
        flat_params = flat_params - op["step_size"] * mhat / (np.sqrt(vhat) + op["eps"])
    return flat_params
