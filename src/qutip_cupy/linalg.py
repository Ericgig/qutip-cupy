from .dense import CuPyDense
from .dia import CuPyDia

import cupy as cp


def inv_cupydense(data):
    """Compute the inverse of a matrix"""
    if not isinstance(data, CuPyDense):
        raise TypeError("expected data in Dense format but got " + str(type(data)))
    if data.shape[0] != data.shape[1]:
        raise ValueError("Cannot compute the matrix inverse" " of a nonsquare matrix")
    return CuPyDense._raw_cupy_constructor(cp.linalg.inv(data._cp))


def matmul_cupydia_cupydense_cupydense(left, right, scale=1):
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            f"incompatible matrix shapes {left.shape} and {right.shape}"
        )
    offsets = left.mat.offsets
    data = left.mat.data

    out = cp.zeros(
        (left.shape[0], right.shape[1]),
        dtype=cp.result_type(left.dtype, right.dtype)
    )
    starts = cp.maximum(offsets, 0)
    ends = cp.minimum(left.shape[1], left.shape[0] + offsets)
    out_starts = cp.maximum(-offsets, 0)
    out_ends = ends - starts + out_starts

    for col in range(right.shape[1]):
        for diag in range(right.data[0]):
            out[col, out_starts[diag]:out_ends[diag]] = (
                data[diag, starts[diag]:ends[diag]]
                * right._cp[col, starts[diag]:ends[diag]]
            )

    return CuPyDense._raw_cupy_constructor(out)


def matmul_cupydia_cupydense(left, right, scale=1.):
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            f"incompatible matrix shapes {left.shape} and {right.shape}"
        )
    offsets = left.offsets
    data = left.data

    out = cp.zeros(
        (left.shape[0], right.shape[1]),
        dtype=cp.result_type(left.dtype, right.dtype)
    )
    starts = cp.maximum(offsets, 0)
    ends = cp.minimum(left.shape[1], left.shape[0] + offsets)
    out_starts = cp.maximum(-offsets, 0)
    out_ends = ends - starts + out_starts

    for col in range(right.shape[1]):
        for diag in range(right.data[0]):
            out[col, out_starts[diag]:out_ends[diag]] = (
                data[diag, starts[diag]:ends[diag]]
                * right[col, starts[diag]:ends[diag]]
                * scale
            )

    return out
