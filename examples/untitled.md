```python
from fundl.datasets import get_simple_sequence
from fundl.layers.rnn import lstm
from fundl.layers import dense
from fundl.losses import _mse_loss
from jax import grad 
from jax.experimental.optimizers import sgd, adam
import jax.numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

N_VOCABULARY = 10
data = get_simple_sequence(N_VOCABULARY)
# data = np.sin(x=np.)
data

# Let's define the model:

def model(p, x):
    out = lstm(p["lstm"], x)
    out = dense(p["dense"], out)
    return out



def mseloss(p, model, x, y):
    yhat = model(p, x)
    return np.mean(_mse_loss(y, yhat))
dloss = grad(mseloss)


from fundl.weights import add_lstm_params, add_dense_params
params = dict()
params = add_lstm_params(params, "lstm", input_dim=8, output_dim=20)
params = add_dense_params(params, "dense", input_dim=20, output_dim=1)




import numpy as onp
# params = add_dense_params(params, "dense", input_dim=3, output_dim=1)
# Reshape data to the structure that an RNN needs.
# We will set it up as sliding windows of 8 slots as input,
# and 1 slot as output.
def sliding_window(sequence, window_size, step=1):
    num_chunks = int((len(sequence) - window_size) / step) + 1
    for i in range(0, num_chunks * step, step):
        yield sequence[i : i + window_size]

stacked_data = onp.vstack(list(sliding_window(data, window_size=9, step=1)))

# Scaling is __very__ important if we are using sigmoids as the activation function.
# In general, we need ReLUs.
stacked_data = scaler.fit_transform(stacked_data)
print(stacked_data)



x = stacked_data[:, :8]
y = stacked_data[:, 8].reshape(-1, 1)

# Debugging
print(model(params, x))


init, update, get_params = adam(step_size=0.001)
print(params)

state = init(params)
for i in range(1000):
    g = dloss(params, model, x, y)
    l = mseloss(params, model, x, y)

    o = model(params, x)
    # print(o)

    state = update(i, g, state)
    params = get_params(state)
    # preds = model(params, x)

    print(i, l)
    # print(preds)
```
