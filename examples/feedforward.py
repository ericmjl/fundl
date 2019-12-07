"""Example Feedforward neural network model."""
from jax import grad, jit
from jax.experimental.optimizers import adam

from fundl.activations import relu
from fundl.layers import dense
from fundl.losses import mseloss, _mse_loss
from fundl.weights import add_dense_params
from sklearn.datasets import make_regression

from time import time

x, y = make_regression(n_samples=1000, n_features=10, n_informative=2)
y = y.reshape(-1, 1)


def model(p, x):
    """Forward neural network model."""
    x = dense(p["dense1"], x, nonlin=relu)
    x = dense(p["dense2"], x)
    return x


params = dict()
params = add_dense_params(params, "dense1", input_dim=10, output_dim=5)
params = add_dense_params(params, "dense2", input_dim=5, output_dim=1)


init, update, get_params = adam(step_size=0.1)

# print(model(params, x))
# print(y)
def mseloss(params, x, y):
    y_hat = model(params, x)
    return _mse_loss(y, y_hat)


print(mseloss(params, x, y))
JIT = True
dloss = grad(mseloss)
if JIT:
    dloss = jit(dloss)

start = time()
state = init(params)
for i in range(1000):
    g = dloss(params, x, y)
    l = mseloss(params, x, y)

    state = update(i, g, state)
    params = get_params(state)

    print(i, l)
print(f"JIT: {JIT}, {time() - start:.2f} seconds")
