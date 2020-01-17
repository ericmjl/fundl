import pytest
from hypothesis import given
from hypothesis import strategies as st

from featurization import SequenceLengthsError, get_embeddings


# We need to generate lists of strings that are of identical length.
# The strings can be of variable letters, but they must be drawn from the a.a. alphabet.
# The length is pre-secified for all sequences, but drawn from a random number generator.
@given(st.data())
def test_get_embeddings(data):

    length = data.draw(st.integers(min_value=1, max_value=10))
    sequences = data.draw(
        st.lists(
            elements=st.text(
                alphabet="MRHKDESTNQCUGPAVIFYWLOXZBJ",
                min_size=length,
                max_size=length,
            ),
            min_size=1,
            # max_size=5,
        )
    )

    embeddings = get_embeddings(sequences)
    assert len(embeddings.shape) == 3
    assert embeddings.shape[0] == len(sequences)
    assert embeddings.shape[1] == length + 1
    assert embeddings.shape[2] == 10


def test_get_embeddings_differengt_lengths():
    sequences = ["MKLV", "MKLNV", "MKLNV"]
    with pytest.raises(SequenceLengthsError):
        get_embeddings(sequences)
