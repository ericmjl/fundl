def add_dense_params(params, name, input_dim, output_dim):
    mu = 0
    sd = 0.1
    params[name] = dict()
    params[name]["w"] = normal(mu, sd, size=(input_dim, output_dim))
    params[name]["b"] = normal(mu, sd, size=(output_dim))
    return params


def add_planar_flow_params(params, name, dim):
    params[name] = dict()
    params[name]["w"] = normal(0, 1, size=(dim, 1))
    params[name]["b"] = normal(0, 1)
    params[name]["u"] = normal(0, 1, size=(dim, 1))
    return params


def add_K_planar_flow_params(params, K, dim):
    Ks = []
    for i in range(K):
        name = f"planar_flow_{i}"
        params = add_planar_flow_params(params, name, dim)
        Ks.append(name)
    return params, Ks
