from time import time

import jax.numpy as np

from fundl.layers.rnn import mlstm1900_batch, mlstm1900
from fundl.utils import sliding_window
from featurization import aa_seq_to_int

start = time()

def load_params():

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

    return params


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
    out = mlstm1900_batch(params, x)
    print("output: ", out)
    print("reps: ", out.mean(axis=0))
    print("output shape: ", out.shape)
    assert out.shape == (x.shape[0], 1900)
    # Check outputs


run_mlstm1900_example()
print(f"Time taken: {time() - start:.2f} seconds")



from featurization import get_embeddings 

def run_mlstm1900_multiple_sequences():
    sequences = ["MKLVNTIAJ", "MKLVNTIAJ", "MKLVNTIAJ", "MKLVNTIAJ", "MKLVNTIAJ"]

    params = load_params()
    x = get_embeddings(sequences)
    out = mlstm1900(x, params)

    print(out.shape)


run_mlstm1900_multiple_sequences()