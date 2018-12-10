from jax.random import normal, PRNGKey

key = PRNGKey(42)
# We standardize all weights to be initialized as a random draw with mean 0,
# scale 0.1
p = dict(loc=0, scale=0.1, key=key)


def add_dense_params(params, name, input_dim, output_dim):
    params[name] = dict()
    params[name]["w"] = normal(size=(input_dim, output_dim), **p)
    params[name]["b"] = normal(size=(output_dim), **p)
    return params


def add_gru_params(params, name, input_dim, output_dim):
    mshape = (n_input, n_output)  # matrix shape
    ashape = (n_output,)  # array shape

    params[name] = dict()
    params[name]["W_z"] = normal(size=mshape, **p)
    params[name]["U_z"] = normal(size=mshape, **p)
    params[name]["b_z"] = normal(size=ashape, **p)

    params[name]["W_r"] = normal(size=mshape, **p)
    params[name]["U_r"] = normal(size=(n_output, n_output), **p)
    params[name]["b_r"] = normal(size=ashape, **p)

    params[name]["W_h"] = normal(size=mshape, **p)
    params[name]["U_h"] = normal(size=(n_output, n_output), **p)
    params[name]["b_h"] = normal(size=ashape, **p)

    return params


def add_planar_flow_params(params, name, dim):
    params[name] = dict()
    params[name]["w"] = normal(size=(dim, 1), **p)
    params[name]["b"] = normal(**p)
    params[name]["u"] = normal(size=(dim, 1), **p)
    return params


def add_K_planar_flow_params(params, K, dim):
    Ks = []
    for i in range(K):
        name = f"planar_flow_{i}"
        params = add_planar_flow_params(params, name, dim)
        Ks.append(name)
    return params, Ks
