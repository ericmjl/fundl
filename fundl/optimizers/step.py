import jax.numpy as np


def adam_ops_init(flat_params):
    adam_ops = {
        "b1": 0.9,
        "b2": 0.999,
        "step_size": 0.001,
        "eps": 1e-8,
        "wd": 0.001,
    }
    adam_ops.update({"m": np.zeros(len(flat_params))})
    adam_ops.update({"v": np.zeros(len(flat_params))})
    return adam_ops


def adam_step(
    op: dict, g: np.ndarray, i, flat_params: np.ndarray, weight_decay=False
):
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
    op["m"] = (1 - op["b1"]) * g + op["b1"] * op[
        "m"
    ]  # First  moment estimate.
    op["v"] = (1 - op["b2"]) * (g ** 2) + op["b2"] * op[
        "v"
    ]  # Second moment estimate.
    mhat = op["m"] / (1 - op["b1"] ** (i + 1))  # Bias correction.
    vhat = op["v"] / (1 - op["b2"] ** (i + 1))
    if weight_decay:
        flat_params = (
            flat_params
            - op["step_size"] * mhat / (np.sqrt(vhat) + op["eps"])
            - op["wd"] * flat_params
        )
    else:
        flat_params = flat_params - op["step_size"] * mhat / (
            np.sqrt(vhat) + op["eps"]
        )
    return flat_params, op
