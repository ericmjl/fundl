import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from featurization import aa_seq_to_int, get_embedding, get_embeddings
from fundl.layers.rnn import mlstm1900
from utils import load_params

params = load_params()


@given(st.data())
@settings(deadline=None)
def test_mlstm1900(data):
    length = data.draw(st.integers(min_value=1, max_value=10))
    sequences = data.draw(
        st.lists(
            elements=st.text(
                alphabet="MRHKDESTNQCUGPAVIFYWLOXZBJ",
                min_size=length,
                max_size=length,
            ),
            min_size=1,
        )
    )

    x = get_embeddings(sequences)
    out = mlstm1900(params, x)
