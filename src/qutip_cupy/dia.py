"""
This module contains the ``CuPyDia`` class and associated function
conversion and specializations for registration with QuTiP's data layer.
"""

import numbers
import bisect

import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np

from qutip.core import data
from qutip.core.data import Dia


class CuPyDia(data.Data):
    """
    """
    def __init__(self, arg, shape=None, copy=True, clean=True, dtype=cp.complex128):
        self.dtype = dtype
        if not isinstance(arg, csp.dia_matrix):
            arg = csp.dia_matrix(arg, shape=shape, copy=copy, dtype=dtype)

        if arg.dtype == self.dtype and not copy and not clean:
            self.mat = arg
        elif clean:
            order = cp.argsort(arg.offsets)
            offsets = cp.zeros_like(arg.offsets)
            data = cp.zeros_like(arg.data)
            for i, idx in enumerate(order):
                offsets[i] = arg.offsets[idx]
                start = max(0, offsets[i])
                end = min(arg.shape[1], arg.shape[0] + offsets[i])
                data[i, start:end] = arg.data[idx, start:end]
            self.mat = csp.dia_matrix((data, offsets), shape=arg.shape, copy=False, dtype=dtype)

        else:
            self.mat = arg.copy()
            self.mat.data = cp.asarray(self.mat.data, dtype=self.dtype)

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

    @property
    def num_diags(self):
        return len(self.mat.offsets)

    def copy(self):
        return CuPyDia(self.mat, copy=True, dtype=self.dtype)

    def to_array(self):
        return cp.asnumpy(self.mat.toarray())

    def conj(self):
        return CuPyDia(
            self.mat._with_data(self.mat.data.conj()),
            copy=False,
            dtype=self.dtype,
        )

    def transpose(self):
        out_offsets = -self.mat.offsets[::-1]
        out_data = cp.zeros(
            (len(self.mat.offsets), self.shape[0]), dtype=self.dtype
        )
        for idx, new_offset in enumerate(out_offsets):
            old_start = max(0, -new_offset)
            new_start = max(0, new_offset)
            new_end = min(self.shape[0], self.shape[1] + new_offset)
            l_diag = new_end - new_start
            out_data[idx, new_start : new_start + l_diag] = \
                self.mat.data[-idx - 1, old_start : old_start + l_diag]
        out = csp.dia_matrix(
            (out_data, out_offsets), shape=self.shape[::-1]
        )

        return CuPyDia(out, copy=False, dtype=self.dtype)

    def adjoint(self):
        return self.transpose().conj()

    def trace(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError(
                f"matrix {self.shape} is not a square matrix."
            )
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


def dia_from_cupydia(cupydia):
    return Dia(
        (cupydia.mat.data.get(), cupydia.mat.offsets.get()),
        shape=cupydia.shape,
    )


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


def mul_cupydia(cpdia, value):
    return CuPyDia(
        cpdia.mat._with_data(cpdia.mat.data * value),
        copy=False,
        dtype=cpdia.dtype
    )


def neg_cupydia(cpdia):
    return CuPyDia(-cpdia.mat, copy=False, dtype=cpdia.dtype)


def _insert_sorted(offsets, val):
    i = bisect.bisect_left(offsets, val)
    if i == len(offsets) or not offsets[i] == val:
        offsets.insert(i, val)


def matmul_cupydia(left, right, scale=1):
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            f"incompatible matrix shapes {left.shape} and {right.shape}"
        )
    out_offsets = []
    for offset_l in left.mat.offsets:
        for offset_r in right.mat.offsets:
            offset = offset_l + offset_r
            if -left.shape[0] < offset < right.shape[1]:
                _insert_sorted(out_offsets, offset)

    out_data = cp.zeros(
        (len(out_offsets), right.shape[1]),
        dtype=cp.result_type(left.dtype, right.dtype)
    )
    out_offsets = cp.array(out_offsets, dtype="i")

    for idx_l, offset_l in enumerate(left.mat.offsets):
        for idx_r, offset_r in enumerate(right.mat.offsets):
            offset = offset_l + offset_r
            if not (-left.shape[0] < offset < right.shape[1]):
                continue

            start_left = (max(0, offset_l) + offset_r)
            start_right = max(0, offset_r)
            start_out = max(0, offset)
            start = max(start_left, start_right, start_out)

            end_left = min(left.shape[1], left.shape[0] + offset_l) + offset_r
            end_right = min(right.shape[1], right.shape[0] + offset_r)
            end_out = min(right.shape[1], left.shape[0] + offset)
            end = min(end_left, end_right, end_out)

            vec_left = left.mat.data[idx_l, max(0, offset_l): min(-1, offset_l)]
            vec_right = right.mat.data[idx_r, max(0, offset_r): min(-1, offset_r)]
            idx = cp.searchsorted(out_offsets, offset)

            out_data[idx, start : end] += (
                left.mat.data[idx_l, start - offset_r : end - offset_r] *
                right.mat.data[idx_r, start : end] *
                scale
            )

    return CuPyDia(
        csp.dia_matrix(
            (out_data, out_offsets),
            shape=(left.shape[0], right.shape[1]),
            copy=False,
        ),
        copy=False,
        dtype=out_data.dtype,
    )


def add_cupydia(left, right, scale=1):
    if left.shape != right.shape:
        raise ValueError(
            "Incompatible shapes for addition of two matrices: "
            f"left={left.shape} and right={right.shape}"
        )
    out_offsets = list(cp.sort(left.mat.offsets))
    for offset in right.mat.offsets:
        _insert_sorted(out_offsets, offset)
    out_data = cp.zeros(
        (len(out_offsets), left.shape[1]),
        dtype=cp.result_type(left.dtype, right.dtype)
    )
    out_offsets = cp.array(out_offsets, dtype="i")

    locs = cp.searchsorted(out_offsets, left.mat.offsets)
    out_data[locs, :] += left.mat.data
    locs = cp.searchsorted(out_offsets, right.mat.offsets)
    out_data[locs, :] += right.mat.data * scale

    return CuPyDia(
        csp.dia_matrix((out_data, out_offsets), shape=left.shape, copy=False),
        copy=False,
        dtype=out_data.dtype,
    )


def iadd_cupydia(left, right, scale=1):
    left.mat += right.mat * scale
    return left


def sub_cupydia(left, right):
    return add_cupydia(left, right, -1)
