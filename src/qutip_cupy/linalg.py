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
    raise NotImplementedError
    return CuPyDense._raw_cupy_constructor(left.mat @ right._cp)
