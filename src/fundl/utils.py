def add_dense_params(params, name, input_dim, output_dim):
    """
    Convenience function for adding parameters to a parameters
    dictionary.
    """
    mu = 0
    sd = 1
    params[name] = dict()
    params[name]["w"] = normal(mu, sd, size=(input_dim, output_dim))
    params[name]["b"] = normal(mu, sd, size=(output_dim))
    return params


def add_gru_params(params, name, input_dim, output_dim):
    mshape = (n_input, n_output)  # matrix shape
    ashape = (n_output,)  # array shape
    p = dict(loc=0, scale=0.1)

    params["name"] = dict()
    params["name"]["W_z"] = normal(size=mshape, **p)
    params["name"]["U_z"] = normal(size=mshape, **p)
    params["name"]["b_z"] = normal(size=ashape, **p)

    params["name"]["W_r"] = normal(size=mshape, **p)
    params["name"]["U_r"] = normal(size=(n_output, n_output), **p)
    params["name"]["b_r"] = normal(size=ashape, **p)

    params["name"]["W_h"] = normal(size=mshape, **p)
    params["name"]["U_h"] = normal(size=(n_output, n_output), **p)
    params["name"]["b_h"] = normal(size=ashape, **p)

    return params


def safe_log(x, eps=1e-10):
    """A logarithmic transform that is numerically safe."""
    return np.log(x + eps)

