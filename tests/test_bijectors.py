from fundl.bijectors import planar_flow
from hypothesis import given
from hypothesis import strategies as st
import autograd.numpy.random as npr

@given(
    st.integers(min_value=2),
    st.integers(min_value=2),
    st.integers(min_value=1),
)
def test_planar_flow(input_cols, output_cols, n_samples):
    params = dict()
    params["w"] = npr.normal(size=(input_cols, output_cols))
    params["b"] = npr.normal(size=(output_cols, 1))
    params["u"] = npr.normal(size=(output_cols, 1))

    z = npr.normal(size=(n_samples, input_cols))

    out = planar_flow(params, z)
    assert out.shape == (n_samples, output_cols)
