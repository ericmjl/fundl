"""Graph node counting task."""

from fundl.datasets import make_graph_counting_dataset
from fundl.utils import pad_graph
import numpy as onp
import networkx as nx
import jax.numpy as np

Gs = make_graph_counting_dataset(n_graphs=1000)
Fs = []
As = []
for G in Gs:
    Fs.append(onp.vstack([d["features"] for n, d in G.nodes(data=True)]))
    As.append(onp.asarray(nx.adjacency_matrix(G).todense()))

largest_graph_size = max([len(G) for G in Gs])

y = np.array([len(G) for G in Gs]).reshape(-1, 1)

for i, (F, A) in enumerate(zip(Fs, As)):
    F, A = pad_graph(F, A, largest_graph_size)
    Fs[i] = F
    As[i] = A

Fs = onp.stack(Fs).astype(float)
As = onp.stack(As).astype(float)

print(Fs.shape)
print(As.shape)


from fundl.weights import add_dense_params

params = dict()
params = add_dense_params(params, name="graph1", input_dim=2, output_dim=5)
params = add_dense_params(params, name="graph2", input_dim=5, output_dim=3)
params = add_dense_params(params, name="dense1", input_dim=3, output_dim=1)


from fundl.layers.graph import mpnn, gather
from fundl.layers import dense
from fundl.activations import relu
from fundl.losses import _mse_loss
from jax import grad


def mseloss(p, model, Fs, As, y):
    yhat = model(p, Fs, As)
    return np.mean(_mse_loss(y, yhat))


def model(params, Fs, As):
    Fs = mpnn(params["graph1"], As, Fs, nonlin=relu)
    Fs = mpnn(params["graph2"], As, Fs, nonlin=relu)
    out = gather(Fs)
    out = dense(params["dense1"], out, nonlin=relu)
    return out


dloss = grad(mseloss)

print(model(params, Fs, As))

from jax.experimental.optimizers import adam

init, update, get_params = adam(step_size=0.005)
print(params)

state = init(params)
for i in range(1000):
    g = dloss(params, model, Fs, As, y)
    l = mseloss(params, model, Fs, As, y)

    state = update(i, g, state)
    params = get_params(state)
    preds = model(params, Fs, As)

    print(i, l)
