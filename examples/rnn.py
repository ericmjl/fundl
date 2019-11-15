import jax.numpy as np
import numpy as onp
import pandas as pd
from jax import grad
from jax.experimental.optimizers import adam, sgd

from fundl.datasets import make_simple_sequence
from fundl.layers import dense
from fundl.layers.rnn import gru, lstm
from fundl.losses import _mse_loss
from fundl.weights import add_dense_params, add_gru_params, add_lstm_params

N_VOCABULARY = 10
data = make_simple_sequence(N_VOCABULARY)


def model(p, x):
    out = lstm(p["lstm"], x)
    # out = lstm(p["lstm2"], out)
    return out


def mseloss(p, model, x, y):
    yhat = model(p, x)
    return np.mean(_mse_loss(y, yhat))


dloss = grad(mseloss)


params = dict()
params = add_lstm_params(params, "lstm", input_dim=8, output_dim=1)
# params = add_lstm_params(params, "lstm2", input_dim=5, output_dim=1)

# Reshape data to the structure that an RNN needs.
# We will set it up as sliding windows of 8 slots as input,
# and 1 slot as output.
def sliding_window(sequence, window_size, step=1):
    num_chunks = int((len(sequence) - window_size) / step) + 1
    for i in range(0, num_chunks * step, step):
        yield sequence[i : i + window_size]


stacked_data = onp.vstack(list(sliding_window(data, window_size=9, step=1)))
print(stacked_data)

x = stacked_data[:, :8]
y = stacked_data[:, 8].reshape(-1, 1)

# Debugging
print(model(params, x))


init, update, get_params = adam(step_size=0.005)
print(params)

state = init(params)
for i in range(200):
    g = dloss(params, model, x, y)
    l = mseloss(params, model, x, y)

    o = model(params, x)

    state = update(i, g, state)
    params = get_params(state)
    preds = model(params, x)

    print(i, l)

df = pd.DataFrame({"preds": preds.ravel(), "actual": y.ravel()})
print(df)
