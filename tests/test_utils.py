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


from fundl.utils import l2_normalize


def test_l2_normalize():
    x = np.array([[3, -3, 5, 4], [4, 5, 3, -3]])

    expected = np.array(
        [
            [3 / 5, -3 / np.sqrt(34), 5 / np.sqrt(34), 4 / 5],
            [4 / 5, 5 / np.sqrt(34), 3 / np.sqrt(34), -3 / 5],
        ],
        dtype=np.float32,
    )

    assert np.allclose(l2_normalize(x, axis=0), expected)
