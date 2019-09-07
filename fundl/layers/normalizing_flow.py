import jax.numpy as np

from ..bijectors import planar_flow, planar_flow_log_det_jacobian


def K_planar_flows(params, z, K):
    """
    :param params: A dictionary of parameters.
    :param z: Samples or transformed samples.
    :param K: a list of flow layer names, e.g. flow_1, flow_2, ...
    """
    log_jacobians = []
    for name in K:
        p = params[name]
        z = planar_flow(p, z)
        j = planar_flow_log_det_jacobian(p, z)
        log_jacobians.append(j)
    return z, np.vstack(log_jacobians).T
