"""Contains specialization functions for dense_cupy. These are the functions that
 are defined outside of qutip/core/data/dense.pyx."""
import bisect
import cupy as cp
import cupyx.scipy.sparse as csp

from .dia import CuPyDia
from .dense_functions import inner_cupydense
from qutip.settings import settings


def _insert_sorted(offsets, val):
    i = bisect.bisect_left(offsets, val)
    if i == len(offsets) or not offsets[i] == val:
        offsets.insert(i, val)


def _check_square_matrix(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "".join(["matrix shape ", str(matrix.shape), " is not square."])
        )


def isherm_cupydia(cp_arr, tol=None):
    # ToDo: check directly without creating a new matrix.
    if cp_arr.shape[0] != cp_arr.shape[1]:
        return False
    return iszero_cupydia(cp_arr - cp_arr.adjoint(), tol=tol)


def iszero_cupydia(cp_arr, tol=None):
    if tol is None:
        tol = settings.core["atol"]
    if len(cp_arr.mat.offsets) == 0:
        return True
    return cp.allclose(cp_arr.mat.data, 0, atol=tol)


def _check_shape_inner(left, right):
    if (left.shape[0] != 1 and left.shape[1] != 1) or right.shape[1] != 1:
        raise ValueError(
            "incompatible matrix shapes " + str(left.shape) + " and " + str(right.shape)
        )


def _check_shape_inner_op(left, op, right):
    left_shape = left.shape[0] == 1 or left.shape[1] == 1
    left_op = (left.shape[0] == 1 and left.shape[1] == op.shape[0]) or (
        left.shape[1] == 1 and left.shape[0] == op.shape[0]
    )
    op_right = op.shape[1] == right.shape[0]
    right_shape = right.shape[1] == 1
    if not (left_shape and left_op and op_right and right_shape):
        raise ValueError(
            "incompatible matrix shapes "
            f"{left.shape}, {op.shape}, {right.shape}."
        )


def inner_op_cupydense_dia_dense(left, op, right, scalar_is_ket=False):
    return inner_cupydense(left, op @ right, scalar_is_ket)


def _kron_sq(left, right):
    nrows = left.shape[0] * right.shape[0]
    ncols = left.shape[1] * right.shape[1]
    out_offsets = []

    for diag_left in range(left.num_diags):
        for diag_right in range(right.num_diags):
            out_diag = (
                left.mat.offsets[diag_left] * right.shape[0]
                + right.mat.offsets[diag_right]
            )
            _insert_sorted(out_offsets, out_diag)

    out_data = cp.zeros(
        (len(out_offsets), ncols),
        dtype=cp.result_type(left.dtype, right.dtype)
    )
    out_offsets = cp.array(out_offsets, dtype="i")

    for diag_left in range(left.num_diags):
        for diag_right in range(right.num_diags):
            offset = (
                left.mat.offsets[diag_left] * right.shape[0]
                + right.mat.offsets[diag_right]
            )
            diag = cp.multiply.outer(
                left.mat.data[diag_left], right.mat.data[diag_right]
            ).ravel(order="C")
            idx = cp.searchsorted(out_offsets, offset)
            out_data[idx, :] += diag

    return CuPyDia(
        (out_data, out_offsets),
        shape=(nrows, ncols),
        copy=False,
        dtype=out_data.dtype,
        clean=False,
    )


def _kron_rec(left, right):
    nrows = left.shape[0] * right.shape[0]
    ncols = left.shape[1] * right.shape[1]
    dtype = cp.result_type(left.dtype, right.dtype)
    out = {}

    delta = right.shape[0] - right.shape[1]
    for diag_left in range(left.num_diags):
        start_left = max(0, left.mat.offsets[diag_left].item())
        end_left = min(left.shape[1], left.shape[0] + left.mat.offsets[diag_left].item())
        for diag_right in range(right.num_diags):
            start_right = max(0, right.mat.offsets[diag_right].item())
            end_right = min(
                right.shape[1], right.shape[0] + right.mat.offsets[diag_right].item()
            )

            for col_left in range(start_left, end_left):
                out_diag = (
                    left.mat.offsets[diag_left] * right.shape[0]
                    + right.mat.offsets[diag_right]
                    - col_left * delta
                ).item()
                data = cp.zeros(ncols, dtype=dtype)
                data[col_left * right.shape[1] : col_left * right.shape[1] + right.shape[1]] = (
                    left.mat.data[diag_left, col_left] * right.mat.data[diag_right]
                )

                if out_diag in out:
                    out[out_diag] = out[out_diag] + data
                else:
                    out[out_diag] = data

    out_offsets = cp.sort(cp.array(list(out.keys())))
    out_data = cp.zeros((len(out_offsets), ncols), dtype=dtype)
    for idx, offset in enumerate(out_offsets):
        out_data[idx, :] = out[offset.item()]

    return CuPyDia(
        (out_data, out_offsets),
        shape=(nrows, ncols),
        copy=False,
        dtype=out_data.dtype,
        clean=False,
    )


def kron_cupydia(left, right):
    if right.shape[0] == right.shape[1]:
        return _kron_sq(left, right)
    else:
        return _kron_rec(left, right)


def frobenius_cupydia(cp_arr):
    return cp.linalg.norm(cp_arr.mat.data).item()


def l2_cupydia(cp_arr):
    if cp_arr.shape[0] != 1 and cp_arr.shape[1] != 1:
        raise ValueError("L2 norm is only defined on vectors")
    return frobenius_cupydia(cp_arr)


def max_cupydia(cp_arr):
    return cp.max(cp.abs(cp_arr.mat.data)).item()


def one_cupydia(cp_arr):
    return cp.linalg.norm(cp_arr.mat.data, ord=1).item()
