from time import time

import jax.numpy as np

from featurization import aa_seq_to_int, get_embeddings
from fundl.layers.rnn import mlstm1900, mlstm1900_batch
from fundl.utils import sliding_window
from utils import load_params

start = time()


def run_mlstm1900_example():
    # Set up an example
    sequence = "MRKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFARYPDHMKQHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    print("sequence length: ", len(sequence))

    sequence = aa_seq_to_int(sequence)[:-1]

    embeddings = np.load("1900_weights/embed_matrix:0.npy")
    x = np.vstack([embeddings[i] for i in sequence])
    print("embedding shape: ", x.shape)

    # x = sliding_window(sequence, size=10)
    params = load_params()
    # Pass through mLSTM1900
    h_final, c_final, out = mlstm1900_batch(params, x)
    print("output: ", out)
    print("reps: ", out.mean(axis=0))
    print("output shape: ", out.shape)
    assert out.shape == (x.shape[0], 1900)


run_mlstm1900_example()
print(f"Time taken: {time() - start:.2f} seconds")


start = time()


def run_mlstm1900_multiple_sequences():
    sequences = [
        "MKLVTITJ",
        "MKLVDIAJ",
        "MKLVTIAJ",
        "MKLRKIAJ",
        "MKLVTIMJ",
    ]

    params = load_params()
    x = get_embeddings(sequences)
    h_final, c_final, out = mlstm1900(params, x)

    print(out.shape)
    print(h_final.shape)
    print(c_final.shape)


run_mlstm1900_multiple_sequences()
print(f"Time taken: {time() - start:.2f} seconds")
