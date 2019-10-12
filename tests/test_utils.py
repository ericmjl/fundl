from fundl.utils import pad_graph
from fundl.datasets import make_graph_counting_dataset
import numpy as np
import networkx as nx 

def test_pad_graph():
    G = make_graph_counting_dataset(1)[0]

    to_size = 15

    F = np.vstack([d["features"] for n, d in G.nodes(data=True)])
    A = np.asarray(nx.adjacency_matrix(G).todense())

    F, A = pad_graph(F, A, to_size=15)
    assert len(F) == len(A)
    assert len(F) == 15
