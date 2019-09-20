"""Dataset generators."""
import numpy as np




def get_simple_sequence(n_vocab, repeat=10):
    data = []
    for i in range(repeat):
        for j in range(n_vocab):
            for k in range(j):
                data.append(j)

    return np.asarray(data, dtype=np.int32)
