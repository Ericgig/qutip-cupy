"""
This module contains the ``CuPyDia`` class and associated function
conversion and specializations for registration with QuTiP's data layer.
"""

import numbers

import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np

from qutip.core import data


class CuPyDia(data.Data):
    """
    """
    def __init__(self, arg, shape=None, copy=True, dtype=cp.complex128):
        self.dtype = dtype
        if isinstance(arg, csp.dia_matrix):
            if arg.dtype == self.dtype and not copy:
                self.mat = arg
            else:
                self.mat = arg.copy()
                self.mat.data = cp.asarray(self.mat.data, dtype=self.dtype)
        else:
            raise NotImplementedError

        if shape is not None:
            if shape != self.mat.shape:
                raise ValueError(
                    f"invalid shape {shape} for input"
                    f" data with shape {self.mat.shape}"
                )

        else:
            shape = self.mat.shape

        if not (
            len(shape) == 2
            and isinstance(shape[0], numbers.Integral)
            and isinstance(shape[1], numbers.Integral)
            and shape[0] > 0
            and shape[1] > 0
        ):
            raise ValueError(
                f"shape must be a 2-tuple of positive ints, but is {shape}"
            )

        super().__init__((shape[0], shape[1]))

    def copy(self):
        return CuPyDia(self.mat, copy=True)

    def to_array(self):
        return cp.asnumpy(self.mat.toarray())

    def conj(self):
        return CuPyDia(self.mat.conj(), copy=False)

    def transpose(self):
        return CuPyDia(self.mat.transpose(), copy=False)

    def adjoint(self):
        return CuPyDia(self.mat.transpose().conj(), copy=False)

    def trace(self):
        loc = cp.where(self.mat.offsets == 0)
        if not loc:
            return 0.j
        return cp.sum(self.mat.data[loc[0]]).item()

    def __imul__(self, other):
        self.mat.__imul__(other)
        return self

    def __itruediv__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        self.mat.__itruediv__(other)
        return self

def zeros(rows, cols, dtype=cp.complex128):
    return CuPyDia(cps.dia_matrix(shape=(rows, cols), dtype=dtype), copy=False)


def identity(dimension, scale=1, dtype=cp.complex128):
    base = cps.eye(shape=dimension, dtype=dtype, format="dia")
    if scale != 1:
        base = base * scale
    return CuPyDia(base, copy=False)

def diags(diagonals, offsets=None, shape=None, dtype=cp.complex128):
    base = csp.diags(diagonals, offsets, shape, dtype=dtype, format="dia")
    return CuPyDia(base, copy=False)


def cupydia_from_dia(dia):
    return CuPyDia(csp.dia_matrix(dia.as_scipy()))


def cupydia_from_cupydense(dia):
    raise NotImplementedError


def cupydense_from_cupydia(cpdia):
    return CuPyDense(cpdia.mat.toarray(), copy=False, dtype=self.dtype)


def adjoint_cupydia(cpdia):
    return cpdia.adjoint()


def conj_cupydia(cpdia):
    return cpdia.conj()


def transpose_cupydia(cpdia):
    return cpdia.transpose()


def trace_cupydia(cpdia):
    return cpdia.trace()


def imul_cupydia(cpdia, value):
    return cpdia.__imul__(value)


def mul_cupydia(cpd_array, value):
    return CuPyDia(cpdia.mat * value, copy=False, dtype=cpdia.dtype)


def neg_cupydia(cpd_array):
    return CuPyDia(cpdia.mat * -1, copy=False, dtype=cpdia.dtype)


def matmul_cupydia(left, right, scale=1, out=None):
    if out is None:
        return CuPyDia(
            left.mat @ right.mat * scale, copy=False, dtype=left.dtype
        )
    else:
        out.mat += (left.mat @ right.mat) * scale
        return out


def add_cupydia(left, right, scale=1):
    return CuPyDia(
        left.mat + right.mat * scale, copy=False, dtype=left.dtype
    )


def iadd_cupydia(left, right, scale=1):
    left.mat += right.mat * scale
    return left


def sub_cupydia(left, right):
    return CuPyDia(
        left.mat - right.mat, copy=False, dtype=left.dtype
    )
