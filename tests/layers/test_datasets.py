import pytest

from fundl.datasets import make_graph_counting_dataset


@pytest.mark.graphs
def test_make_graph_counting_dataset():
    Gs = make_graph_counting_dataset(10)
    assert len(Gs) == 10
