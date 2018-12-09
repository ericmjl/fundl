def planar_flow(params, z):
    """
    Transforms z using planar transformations.

    :param z: Samples or transformed samples.
    :param params: A dictionary of tunable parameters.
    """
    a = (
        np.dot(params["w"].T, z.T) + params["b"]
    )  # (w: (dim, 1), z: (n, dim), b: scalar, a: (1, n))
    return (
        z + (params["u"] * np.tanh(a)).T
    )  # (u: (dim, 1), a: (1, n), z: (n, dim))


def planar_flow_log_det_jacobian(params, z):
    """
    log of abs of determinant of planar flow jacobians

    :param params: A dictionary of parameters.
    """
    a = np.dot(params["w"].T, z.T) + params["b"]
    psi = dtanh(a) * params["w"]
    det_grad = 1 + np.dot(params["u"].T, psi)
    return safe_log(np.abs(det_grad))
