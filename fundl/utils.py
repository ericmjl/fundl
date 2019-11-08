import jax.numpy as np
import numpy as onp


def safe_log(x, eps=1e-10):
    """A logarithmic transform that is numerically safe."""
    return np.log(x + eps)


def ndims(x):
    """
    Return the ndims of a tensor.
    """
    return len(x.shape)


def pad_graph(F, A, to_size: int):
    """Pad F and A matrices with zeros to fit graph size."""
    pad_size = to_size - len(F)
    F = onp.pad(F, [(0, pad_size), (0, 0)])
    A = onp.pad(A, [(0, pad_size), (0, pad_size)])
    return F, A


def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal.

    Taken from:
    https://gist.github.com/nils-werner/9d321441006b112a4b116a8387c2280c

    No credit taken for this implementation, h/t @nils-werner! :D

    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.


    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.

    Notes
    -----

    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.

    Examples
    --------

    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])

    See Also
    --------
    pieces : Calculate number of pieces available by sliding

    """
    if axis >= data.ndim:
        raise ValueError("Axis value out of range")

    if stepsize < 1:
        raise ValueError("Stepsize may not be zero or negative")

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = onp.floor(
        data.shape[axis] / stepsize - size / stepsize + 1
    ).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = onp.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided


def l2_normalize(arr, axis, epsilon=1e-12):
    """
    L2 normalize along a particular axis.

    Doc taken from tf.nn.l2_normalize:
    https://www.tensorflow.org/api_docs/python/tf/math/l2_normalize

    output = x / (
        sqrt(
            max(
                sum(x**2),
                epsilon
            )
        )
    )
    """
    sq_arr = np.power(arr, 2)
    square_sum = np.sum(sq_arr, axis=axis, keepdims=True)
    max_weights = np.maximum(square_sum, epsilon)
    return np.divide(arr, np.sqrt(max_weights))
