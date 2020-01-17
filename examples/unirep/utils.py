from pathlib import Path

import jax.numpy as np

this_dir = Path(__file__).resolve().parent


def load_params():

    params = dict()
    params["gh"] = np.load(this_dir / "1900_weights/rnn_mlstm_mlstm_gh:0.npy")
    params["gmh"] = np.load(
        this_dir / "1900_weights/rnn_mlstm_mlstm_gmh:0.npy"
    )
    params["gmx"] = np.load(
        this_dir / "1900_weights/rnn_mlstm_mlstm_gmx:0.npy"
    )
    params["gx"] = np.load(this_dir / "1900_weights/rnn_mlstm_mlstm_gx:0.npy")

    params["wh"] = np.load(this_dir / "1900_weights/rnn_mlstm_mlstm_wh:0.npy")
    params["wmh"] = np.load(
        this_dir / "1900_weights/rnn_mlstm_mlstm_wmh:0.npy"
    )
    params["wmx"] = np.load(
        this_dir / "1900_weights/rnn_mlstm_mlstm_wmx:0.npy"
    )
    params["wx"] = np.load(this_dir / "1900_weights/rnn_mlstm_mlstm_wx:0.npy")

    params["b"] = np.load(this_dir / "1900_weights/rnn_mlstm_mlstm_b:0.npy")

    return params
