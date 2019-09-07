import autograd.numpy.random as npr
from hypothesis import given
from hypothesis import strategies as st

from fundl.bijectors import planar_flow


@given(
    st.integers(min_value=2, max_value=20),
    st.integers(min_value=1, max_value=20),
)
def test_planar_flow(input_cols, n_samples):
    params = dict()
    params["w"] = npr.normal(size=(input_cols, 1))
    params["b"] = npr.normal(size=(1, n_samples))
    params["u"] = npr.normal(size=(input_cols, 1))

    z = npr.normal(size=(n_samples, input_cols))

    out = planar_flow(params, z)
    assert out.shape == (n_samples, input_cols)
