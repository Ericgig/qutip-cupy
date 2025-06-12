"""Contains specialization functions for dense_cupy. These are the functions that
 are defined outside of qutip/core/data/dense.pyx."""
import cupy as cp

from .dia import CuPyDia


def reshape_cupydia(cp_arr, n_rows_out, n_cols_out):
    return CuPyDia(
        cp_arr.mat.reshape((n_rows_out, n_cols_out)),
        dtype = cp_arr.dtype,
        copy = False,
    )


def column_stack_cupydia(cp_arr):
    return CuPyDia(
        cp_arr.mat.reshape((-1, 1), order="F"),
        dtype = cp_arr.dtype,
        copy = False,
    )


def column_unstack_cupydia(cp_arr, rows):
    if cp_arr.shape[1] != 1:
        raise ValueError("input is not a single column")
    if rows < 1:
        raise ValueError("rows must be a positive integer")
    if cp_arr.shape[0] % rows:
        raise ValueError("number of rows does not divide into the shape")
    return CuPyDia(
        cp_arr.mat.reshape((rows, -1), order="F"),
        dtype = cp_arr.dtype,
        copy = False,
    )


def _check_square_matrix(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "".join(["matrix shape ", str(matrix.shape), " is not square."])
        )

def isherm_cupydia(cp_arr, tol):
    return iszero_cupydia(cp_arr - cp_arr.adjoint(), tol=tol)


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


def kron_cupydia(left, right):
    return CuPyDia(csp.kron(left.mat, right.mat), copy=False, dtype=left.dtype)


def frobenius_cupydia(cp_arr):
    # TODO: Expose CUBLAS' dznrm2 (like QuTiP does) and test if it is faster
    return cp.linalg.norm(cp_arr.mat.data).item()


def l2_cupydia(cp_arr):
    if cp_arr.shape[0] != 1 and cp_arr.shape[1] != 1:
        raise ValueError("L2 norm is only defined on vectors")
    return frobenius_cupydia(cp_arr)


def max_cupydia(cp_arr):
    return cp.max(cp.abs(cp_arr.mat.data)).item()


def one_cupydia(cp_arr):
    return cp.linalg.norm(cp_arr.mat.data, ord=1).item()
