"""Dataset generators."""
import networkx as nx
import numpy as np


def make_simple_sequence(n_vocab, repeat=10):
    data = []
    for i in range(repeat):
        for j in range(n_vocab):
            for k in range(j):
                data.append(j)

    return np.asarray(data, dtype=np.int32)


def make_graph_counting_dataset(n_graphs):
    """
    Graph dataset generator.

    Generates a n_graphs,
    each with a number of nodes,
    and a set of features attached to each node
    as node attributes.

    The features are stored under the key "feats";
    the first slot consistently has the value "1" on it,
    while the second slot has a random integer
    drawn from a standard Normal distribution.
    """
    graphs = []
    for i in range(n_graphs):
        n_nodes = np.random.randint(5, 11)
        G = nx.random_regular_graph(d=4, n=n_nodes)
        for n in G.nodes():
            G.node[n]["features"] = np.array([1, np.random.normal()])
        graphs.append(G)
    return graphs
