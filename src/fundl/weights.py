from jax.random import normal, PRNGKey, split

key = PRNGKey(42)


def add_dense_params(params, name, input_dim, output_dim):
    params[name] = dict()
    params[name]["w"] = normal(split(key)[0], (input_dim, output_dim))
    params[name]["b"] = normal(split(key)[0], (output_dim,))
    return params


def add_gru_params(params, name, input_dim, output_dim):
    mshape = (n_input, n_output)  # matrix shape
    ashape = (n_output,)  # array shape

    params[name] = dict()
    params[name]["W_z"] = normal(key, mshape)
    params[name]["U_z"] = normal(key, mshape)
    params[name]["b_z"] = normal(key, ashape)

    params[name]["W_r"] = normal(key, mshape)
    params[name]["U_r"] = normal(key, (n_output, n_output))
    params[name]["b_r"] = normal(key, ashape)

    params[name]["W_h"] = normal(key, mshape)
    params[name]["U_h"] = normal(key, (n_output, n_output))
    params[name]["b_h"] = normal(key, ashape)

    return params


def add_planar_flow_params(params, name, dim):
    params[name] = dict()
    params[name]["w"] = normal(key, (dim, 1))
    params[name]["b"] = normal(key,)
    params[name]["u"] = normal(key, (dim, 1))
    return params


def add_K_planar_flow_params(params, K, dim):
    Ks = []
    for i in range(K):
        name = f"planar_flow_{i}"
        params = add_planar_flow_params(params, name, dim)
        Ks.append(name)
    return params, Ks
