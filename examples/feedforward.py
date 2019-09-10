from fundl.layers import dense
from fundl.activations import relu
from fundl.weights import add_dense_params
from fundl.losses import mseloss

from jax.experimental.optimizers import adam
from jax import grad
from sklearn.datasets import make_regression


x, y = make_regression(n_samples=1000, n_features=10, n_informative=2)
y = y.reshape(-1, 1)


def model(p, x):
    x = dense(p["dense1"], x, nonlin=relu)
    x = dense(p["dense2"], x)
    return x


params = dict()
params = add_dense_params(params, "dense1", input_dim=10, output_dim=5)
params = add_dense_params(params, "dense2", input_dim=5, output_dim=1)


init, update, get_params = adam(step_size=0.1)

# print(model(params, x))
# print(y)
print(mseloss(params, model, x, y))
dloss = grad(mseloss)

state = init(params)
for i in range(1000):
    g = dloss(params, model, x, y)
    l = mseloss(params, model, x, y)

    state = update(i, g, state)
    params = get_params(state)

    print(i, l)
