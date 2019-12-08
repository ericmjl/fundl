from time import time

import jax.numpy as np

from fundl.layers.rnn import mlstm1900_batch
from fundl.utils import sliding_window

start = time()

aa_to_int = {
    "M": 1,
    "R": 2,
    "H": 3,
    "K": 4,
    "D": 5,
    "E": 6,
    "S": 7,
    "T": 8,
    "N": 9,
    "Q": 10,
    "C": 11,
    "U": 12,
    "G": 13,
    "P": 14,
    "A": 15,
    "V": 16,
    "I": 17,
    "F": 18,
    "Y": 19,
    "W": 20,
    "L": 21,
    "O": 22,  # Pyrrolysine
    "X": 23,  # Unknown
    "Z": 23,  # Glutamic acid or GLutamine
    "B": 23,  # Asparagine or aspartic acid
    "J": 23,  # Leucine or isoleucine
    "start": 24,
    "stop": 25,
}


def aa_seq_to_int(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return [24] + [aa_to_int[a] for a in s] + [25]


one_hots = np.eye(26)


def run_mlstm1900_example():
    # Set up an example
    sequence = "MRKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFARYPDHMKQHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    print("sequence length: ", len(sequence))

    sequence = aa_seq_to_int(sequence)[:-1]

    embeddings = np.load("1900_weights/embed_matrix:0.npy")
    x = np.vstack([embeddings[i] for i in sequence])
    print("embedding shape: ", x.shape)

    # x = sliding_window(sequence, size=10)
    params = dict()
    params["gh"] = np.load("1900_weights/rnn_mlstm_mlstm_gh:0.npy")
    params["gmh"] = np.load("1900_weights/rnn_mlstm_mlstm_gmh:0.npy")
    params["gmx"] = np.load("1900_weights/rnn_mlstm_mlstm_gmx:0.npy")
    params["gx"] = np.load("1900_weights/rnn_mlstm_mlstm_gx:0.npy")

    params["wh"] = np.load("1900_weights/rnn_mlstm_mlstm_wh:0.npy")
    params["wmh"] = np.load("1900_weights/rnn_mlstm_mlstm_wmh:0.npy")
    params["wmx"] = np.load("1900_weights/rnn_mlstm_mlstm_wmx:0.npy")
    params["wx"] = np.load("1900_weights/rnn_mlstm_mlstm_wx:0.npy")

    params["b"] = np.load("1900_weights/rnn_mlstm_mlstm_b:0.npy")

    # Pass through mLSTM1900
    out = mlstm1900_batch(params, x)
    print("output: ", out)
    print("reps: ", out.mean(axis=0))
    print("output shape: ", out.shape)
    assert out.shape == (x.shape[0], 1900)
    # Check outputs


run_mlstm1900_example()
print(f"Time taken: {time() - start:.2f} seconds")
