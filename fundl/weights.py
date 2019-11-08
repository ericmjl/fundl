from jax.random import PRNGKey, normal, split

key = PRNGKey(42)


def add_dense_params(params, name, input_dim, output_dim):
    params[name] = dict()
    params[name]["w"] = normal(split(key)[0], (input_dim, output_dim)) * 0.01
    params[name]["b"] = normal(split(key)[0], (output_dim,)) * 0.01
    return params


def add_gru_params(params, name, input_dim, output_dim):
    mshape = (input_dim, output_dim)  # matrix shape
    ashape = (output_dim,)  # array shape

    params[name] = dict()
    params[name]["W_z"] = normal(split(key)[0], mshape) * 0.01
    params[name]["U_z"] = normal(split(key)[0], mshape) * 0.01
    params[name]["b_z"] = normal(split(key)[0], ashape) * 0.01

    params[name]["W_r"] = normal(split(key)[0], mshape) * 0.01
    params[name]["U_r"] = (
        normal(split(key)[0], (output_dim, output_dim)) * 0.01
    )
    params[name]["b_r"] = normal(split(key)[0], ashape) * 0.01

    params[name]["W_h"] = normal(split(key)[0], mshape) * 0.01
    params[name]["U_h"] = (
        normal(split(key)[0], (output_dim, output_dim)) * 0.01
    )
    params[name]["b_h"] = normal(split(key)[0], ashape) * 0.01

    return params


def add_planar_flow_params(params, name, dim):
    params[name] = dict()
    params[name]["w"] = normal(split(key)[0], shape=(dim, 1)) * 0.01
    params[name]["b"] = normal(split(key)[0], shape=(1,)) * 0.01
    params[name]["u"] = normal(split(key)[0], shape=(dim, 1)) * 0.01
    return params


def add_K_planar_flow_params(params, K, dim):
    Ks = []
    for i in range(K):
        name = f"planar_flow_{i}"
        params = add_planar_flow_params(params, name, dim)
        Ks.append(name)
    return params, Ks


def add_lstm_params(params, name, input_dim, output_dim):
    mshape = (output_dim, input_dim + output_dim)  # matrix shape
    ashape = (output_dim,)  # array shape

    params[name] = dict()
    params[name]["W_i"] = normal(split(key)[0], mshape) * 0.01
    params[name]["b_i"] = normal(split(key)[0], ashape) * 0.01

    params[name]["W_c"] = normal(split(key)[0], mshape) * 0.01
    params[name]["b_c"] = normal(split(key)[0], ashape) * 0.01

    params[name]["W_f"] = normal(split(key)[0], mshape) * 0.01
    params[name]["b_f"] = normal(split(key)[0], ashape) * 0.01

    params[name]["W_o"] = normal(split(key)[0], mshape) * 0.01
    params[name]["b_o"] = normal(split(key)[0], ashape) * 0.01

    return params


def add_mlstm1900_params(params, name, input_dim, output_dim):
    params[name] = dict()
    params[name]["wmx"] = normal(split(key)[0], (input_dim, output_dim))
    params[name]["wmh"] = normal(split(key)[0], (output_dim, output_dim))
    params[name]["wx"] = normal(split(key)[0], (input_dim, output_dim * 4))
    params[name]["wh"] = normal(split(key)[0], (output_dim, output_dim * 4))

    params[name]["gmx"] = normal(split(key)[0], (output_dim,))
    params[name]["gmh"] = normal(split(key)[0], (output_dim,))
    params[name]["gx"] = normal(split(key)[0], (output_dim * 4,))
    params[name]["gh"] = normal(split(key)[0], (output_dim * 4,))

    params[name]["b"] = normal(split(key)[0], (output_dim * 4,))
    return params
