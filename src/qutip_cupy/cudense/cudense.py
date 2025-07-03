try:
    import cuquantum.densitymat as cudense
    MultidiagonalOperator = cudense.MultidiagonalOperator
    DenseOperator = cudense.DenseOperator
    OperatorTerm = cudense.OperatorTerm
    ScalarCallbackCoefficient = cudense._internal.callbacks.ScalarCallbackCoefficient

except ImportError:
    class _Missing:
        ...

    cudense = None
    MultidiagonalOperator = _Missing
    DenseOperator = _Missing
    OperatorTerm = _Missing
    ScalarCallbackCoefficient = _Missing


from enum import Enum
from typing import NamedTuple, Any
import itertools
import numpy as np

from qutip.core.data import Data
from qutip.core import data as _data

from ..dense import CuPyDense

# __all__ = ["CuOperator"]


def _compare_hilbert(left, right, return_shifts=False):

    def _set(hilbert, idx, shift):
        if len(hilbert) > idx:
            shift[idx] = []
            return hilbert[idx]
        return None

    ptr_l = 0
    ptr_r = 0
    out_hilbert = []
    shifts_left = {}
    shifts_right = {}

    size_l = _set(left, ptr_l, shifts_left)
    size_r = _set(right, ptr_r, shifts_right)

    while ptr_l < len(left) and ptr_r < len(right):
        shifts_left[ptr_l].append(len(out_hilbert))
        shifts_right[ptr_r].append(len(out_hilbert))

        if abs(size_l) == abs(size_r):
            out_hilbert.append(max(size_l, size_r))
            ptr_l += 1
            ptr_r += 1
            size_l = _set(left, ptr_l, shifts_left)
            size_r = _set(right, ptr_r, shifts_right)

        elif -size_l > abs(size_r):
            if size_l % size_r != 0:
                return False
            out_hilbert.append(size_r)
            size_l = -abs(size_l // size_r)
            ptr_r += 1
            size_r = _set(right, ptr_r, shifts_right)

        elif abs(size_l) < -size_r:
            if size_r % size_l != 0:
                return False
            out_hilbert.append(size_l)
            size_r = -abs(size_r // size_l)
            ptr_l += 1
            size_l = _set(left, ptr_l, shifts_left)

        else:
            return False

        if size_l is None and size_r is None:
            break
        elif size_l is None or size_r is None:
            return False

    if return_shifts:
        return out_hilbert, shifts_left, shifts_right
    else:
        return out_hilbert


def _apply_transformation(oper, transform):
    if transform == Transform.DIRECT:
        out = oper
    if transform == Transform.CONJ:
        if isinstance(oper, _data.Data):
            out = oper.conj()
        elif isinstance(oper, DenseOperator):
            out = DenseOperator(oper.to_array().conj())
        elif isinstance(oper, MultidiagonalOperator):
            out = MultidiagonalOperator(oper.data.conj(), oper.offsets)
    if transform == Transform.TRANSPOSE:
        if isinstance(oper, _data.Data):
            out = oper.transpose()
        elif isinstance(oper, DenseOperator):
            out = DenseOperator(oper.data.T)
        elif isinstance(oper, MultidiagonalOperator):
            out = MultidiagonalOperator(oper.data, [-i for i in oper.offsets])
    if transform == Transform.ADJOINT:
        if isinstance(oper, _data.Data):
            out = oper.adjoint()
        elif isinstance(oper, (DenseOperator, MultidiagonalOperator)):
            out = oper.dag()
    return out


def _oper_to_ElementaryOperator(oper, hilbert_dims, hilbert_idx):
    N = len(hilbert_idx)
    shape = (hilbert_dims[idx] for idx in hilbert_idx)
    if isinstance(oper, (DenseOperator, MultidiagonalOperator)):
        if N != 1 and isinstance(oper, MultidiagonalOperator):
            raise ValueError("MultidiagonalOperator on multiple hilbert spaces")
        if N == 1 and oper.shape[0] != shape[0]:
            raise ValueError("Operator shape does not match hilbert spaces")
        if list(oper.shape[:-1]) != list(shape + shape):
            raise ValueError("Operator shape does not match hilbert spaces")
        out = oper
    elif isinstance(oper, _data.Dia) and N == 1:
        dia_matrix = dia.as_scipy()
        offsets = list(dia_matrix.offsets)
        data = np.zeros((dia_matrix.shape[0], len(offsets)), dtype=complex)
        for i, offset in enumerate(offsets):
            end = None if offset == 0 else -abs(offset)
            data[:end, i] = dia_matrix.diagonal(offset)
        out = MultidiagonalOperator(data, offsets)
    else:
        out = DenseOperator(mat.to_array().reshape(shape + shape))
    return out


###############################################################################
###############################################################################


class Transform(Enum):
    DIRECT = 0
    CONJ = 1
    TRANSPOSE = 2
    ADJOINT = 3


conj_transform = {
    Transform.DIRECT : Transform.CONJ,
    Transform.CONJ : Transform.DIRECT,
    Transform.TRANSPOSE : Transform.ADJOINT,
    Transform.ADJOINT : Transform.TRANSPOSE,
}

trans_transform = {
    Transform.DIRECT : Transform.TRANSPOSE,
    Transform.CONJ : Transform.ADJOINT,
    Transform.TRANSPOSE : Transform.DIRECT,
    Transform.ADJOINT : Transform.CONJ,
}

adjoint_transform = {
    Transform.DIRECT : Transform.ADJOINT,
    Transform.CONJ : Transform.TRANSPOSE,
    Transform.TRANSPOSE : Transform.CONJ,
    Transform.ADJOINT : Transform.DIRECT,
}


class ProdTerm(NamedTuple):
    operator: Any
    hilbert: tuple[int]
    transform: Transform


class Term(NamedTuple):
    prod_terms: list[ProdTerm]
    factor: complex


def _has_dual(arg):
    for duals in zip(self.base.duals):
        for dual in zip(duals):
            if dual:
                return True
    return False


class CuOperator(Data):
    terms: list
    hilbert_dims: tuple

    def __init__(self, arg=None, shape=None, copy=True, hilbert_dims=None):
        self.terms = []
        self.hilbert_dims = ()
        oper_shape = None

        if arg is None:
            self.hilbert_dims = (-shape[0], )
            oper_shape = shape

        elif isinstance(arg, MultidiagonalOperator):
            oper_shape = arg.shape
            arg = arg.copy() if copy else arg
            self.terms.append(
                Term([ProdTerm(arg, (0,), Transform.DIRECT)], 1.+0j)
            )
            self.hilbert_dims = (arg.shape[0], )

        elif isinstance(arg, DenseOperator):
            oper_shape = arg.shape
            arg = arg.copy() if copy else arg
            self.terms.append(
                Term([ProdTerm(arg, (0,), Transform.DIRECT)], 1.+0j)
            )
            self.hilbert_dims = tuple(arg.data.shape[:arg.data.shape // 2])

        elif isinstance(arg, OperatorTerm):
            if hilbert_dims is None:
                hilbert_dims = arg.hilbert_space_dims
            if hilbert_dims is None:
                raise ValueError(...)

            oper_shape = (N, N)

            has_dual = _has_dual(arg)
            N = len(hilbert_dims)
            for terms, modes, duals, coeff in zip(arg.terms, arg.modes, arg.duals, arg._coefficients):
                if not isinstance(coeff, ScalarCallbackCoefficient):
                    raise ValueError(...)
                terms = Term([], factor=coeff._static_coeff)

                for term, mode, dual in zip(terms, modes, duals):
                    term = term.copy() if copy else term
                    if has_dual and not dual:
                        mode = tuple(i + N for i in mode)
                    if has_dual and dual:
                        terms.prod_terms.append(ProdTerm(term, mode, Transform.TRANSPOSE))
                    else:
                        terms.prod_terms.append(ProdTerm(term, mode, Transform.DIRECT))

            if has_dual:
                self.hilbert_dims = hilbert_dims + hilbert_dims
            else:
                self.hilbert_dims = hilbert_dims
            hilbert_dims = None

        elif isinstance(arg, Data) and not isinstance(arg, CuOperator):
            oper_shape = arg.shape
            arg = arg.copy() if copy else arg
            self.terms.append(
                Term([ProdTerm(arg, (0,), Transform.DIRECT)], 1.+0j)
            )
            self.hilbert_dims = (-arg.shape[0], )

        else:
            raise TypeError(...)

        if shape and shape != oper_shape:
            raise ValueError(...)
        if oper_shape[0] != oper_shape[1]:
            raise ValueError(...)
        if abs(np.prod(self.hilbert_dims)) != oper_shape[0]:
            raise ValueError(...)

        super().__init__(oper_shape)

        if hilbert_dims is not None:
            self._update_hilbert(hilbert_dims)

    def _update_hilbert(self, new):
        matched = _compare_hilbert(self.hilbert_dims, new, return_shifts=True)
        if not matched:
            raise ValueError(...)

        new_hilbert, shifts, _ = matched
        self.hilbert_dims = tuple(new_hilbert)
        new_terms = []
        for term in self.terms:
            copy_term = Term([], factor=term.factor)
            for pterm in term.prod_terms:
                new_mode = []
                for i in pterm.hilbert:
                    new_mode += shifts[i]
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator,
                    tuple(new_mode),
                    pterm.transform,
                ))
            new_terms.append(copy_term)
        self.terms = new_terms

    def copy(self, shallow=False):
        new = CuOperator(shape=self.shape, hilbert_dims=self.hilbert_dims)
        for term in self.terms:
            copy_term = Term([], factor=term.factor)
            for pterm in term.prod_terms:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator.copy() if not shallow else pterm.operator,
                    pterm.hilbert,
                    pterm.transform,
                ))
            new.terms.append(copy_term)
        return new

    def to_array(self):
        hilbert = self.hilbert_dims
        out = np.zeros(self.shape, dtype=complex)

        for A, term in enumerate(self.terms):
            termmat = np.eye(self.shape[0], dtype=complex) * term.factor
            for B, prod_term in enumerate(term.prod_terms):
                mat = _apply_transformation(prod_term.operator, prod_term.transform)
                mat = mat.to_array()

                try:
                    # if cupy array, get numpy
                    mat = mat.get()
                except:
                    pass
                if len(mat.shape) > 2:
                    mat = mat.reshape(oper.shape + (-1,))[:, :, 0]

                idxs = list(range(len(hilbert)))
                sizes = []
                for i in prod_term.hilbert:
                    sizes.append(abs(hilbert[idxs.pop(i)]))
                for i in idxs:
                    N = abs(hilbert[i])
                    mat = np.kron(mat, np.eye(N))
                    sizes.append(N)
                mat = _data.permute.dimensions(
                    _data.Dense(mat),
                    sizes,
                    list(prod_term.hilbert) + idxs,
                    dtype=_data.Dense,
                ).to_array()
                termmat = mat @ termmat
            out += termmat
        return out

    def conj(self):
        new = CuOperator(shape=self.shape, hilbert_dims=self.hilbert_dims)
        for term in self.terms:
            copy_term = Term([], factor=term.factor)
            for pterm in term.prod_terms:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator.copy(),
                    pterm.hilbert,
                    conj_transform[pterm.transform],
                ))
            new.terms.append(copy_term)
        return new

    def transpose(self):
        new = CuOperator(shape=self.shape, hilbert_dims=self.hilbert_dims)
        for term in self.terms:
            copy_term = Term([], factor=term.factor)
            for pterm in term.prod_terms[::-1]:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator.copy(),
                    pterm.hilbert,
                    trans_transform[pterm.transform],
                ))
            new.terms.append(copy_term)
        return new

    def adjoint(self):
        new = CuOperator(shape=self.shape, hilbert_dims=self.hilbert_dims)
        for term in self.terms:
            copy_term = Term([], factor=term.factor)
            for pterm in term.prod_terms[::-1]:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator.copy(),
                    pterm.hilbert,
                    adjoint_transform[pterm.transform],
                ))
            new.terms.append(copy_term)
        return new

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        if not isinstance(other, CuOperator):
            if isinstance(other, Data):
                return _data.add(self, other)
            return NotImplemented

        if not _compare_hilbert(self.hilbert_dims, other.hilbert_dims):
            raise ValueError()
        new = self.copy()
        new._update_hilbert(other.hilbert_dims)
        other = other.copy()
        new.terms += other.terms

        return new

    def __sub__(self, other):
        if not isinstance(other, CuOperator):
            if isinstance(other, Data):
                return _data.sub(self, other)
            return NotImplemented

        if not _compare_hilbert(self.hilbert_dims, other.hilbert_dims):
            raise ValueError()
        return self + -other

    def __mul__(self, other):
        new = CuOperator(shape=self.shape, hilbert_dims=self.hilbert_dims)
        for term in self.terms:
            copy_term = Term([], factor=term.factor * other)
            for pterm in term.prod_terms:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator.copy(),
                    pterm.hilbert,
                    pterm.transform,
                ))
            new.terms.append(copy_term)
        return new

    def __div__(self, other):
        return self * (1 / other)

    def __matmul__(self, other):
        if not isinstance(other, CuOperator):
            if isinstance(other, Data):
                return _data.matmul(self, other)
            return NotImplemented

        if not _compare_hilbert(self.hilbert_dims, other.hilbert_dims):
            raise ValueError()
        left = self.copy()
        left._update_hilbert(other.hilbert_dims)
        right = other.copy()
        right._update_hilbert(self.hilbert_dims)

        new = CuOperator(shape=self.shape, hilbert_dims=left.hilbert_dims)

        for term_left, term_right in itertools.product(left.terms, right.terms):
            new.terms.append(
                Term(
                    term_left.prod_terms + term_right.prod_terms,
                    term_left.factor * term_right.factor,
                )
            )

        return new

    def to_OperatorTerm(self, dual=False, copy=True):
        out = 0
        if not dual:
            for term in self.terms:
                cuterm = 1
                for pterm in term.prod_terms:
                    oper = _apply_transformation(pterm.operator, pterm.transform)
                    oper = _oper_to_ElementaryOperator(oper, copy)
                    cuterm = OperatorTerm((oper, pterm.hilbert)) * cuterm
                out += cuterm * term.factor
        else:
            raise NotImplementedError
        return out


def CuOperator_from_Dia():
    return CuOperator(mat)


def CuOperator_from_Dense(mat):
    return CuOperator(mat)


def CuOperator_from_CuDense(mat):
    return CuOperator(mat)


def Dense_from_CuOperator(mat):
    print("converting to Dense")
    return _data.Dense(mat.to_array())


def identity_CuOperator(dimension, scale=1):
    new = CuOperator(shape=(dimension, dimension))
    new.terms.append(Term([], complex(scale)))
    return new


def zeros_CuOperator(rows, cols):
    return CuOperator(shape=(rows, cols))


def diags_CuOperator(diagonals, offsets=None, shape=None):
    return CuOperator(_data.dia.diags(diagonals, offsets, shape))


def kron_CuOperator(left, right):
    N = len(left.hilbert_dims)
    S = left.shape[0] * right.shape[0]
    new_hilbert = left.hilbert_dims + right.hilbert_dims

    left_ext = CuOperator(shape=(S, S), hilbert_dims=new_hilbert)
    for term in left.terms:
        copy_term = Term([], factor=term.factor)
        for pterm in term.prod_terms:
            copy_term.prod_terms.append(ProdTerm(
                pterm.operator.copy(),
                pterm.hilbert,
                pterm.transform,
            ))
        left_ext.terms.append(copy_term)

    right_shifted = CuOperator(shape=(S, S), hilbert_dims=new_hilbert)
    for term in right.terms:
        copy_term = Term([], factor=term.factor)
        for pterm in term.prod_terms:
            copy_term.prod_terms.append(ProdTerm(
                pterm.operator.copy(),
                tuple(i + N for i in pterm.hilbert),
                pterm.transform,
            ))
        right_shifted.terms.append(copy_term)

    return left_ext @ right_shifted


def dimensions_CuOperator(matrix, hilbert, order):
    """
    Reorder the tensor-product structure of a matrix, assuming that the
    underlying structure is defined by `dimensions`.  For a separable system,
    this function produces a matrix which is equivalent to having performed
    `kron` in a different order on the separable parts.

    For example if `a`, `b` and `c` are matrices with sizes 2, 3 and 4
    respectively, then
        kron(kron(c, a), b) == permute.dimensions(kron(kron(a, b), c),
                                                  [2, 3, 4],
                                                  [1, 2, 0])
    In other words, the inputs to `kron` are reordered so that input `n` moves
    to position `order[n]`.
    """
    assert _compare_hilbert(matrix.hilbert_dims, hilbert)
    new = CuOperator(shape=matrix.shape)
    permutation = np.argsort(order)
    new.hilbert_dims = tuple(matrix.hilbert_dims[i] for i in permutation)

    for term in matrix.terms:
        copy_term = Term([], factor=term.factor)
        for pterm in term.prod_terms:
            copy_term.prod_terms.append(ProdTerm(
                pterm.operator.copy(),
                tuple(order[mode] for mode in pterm.hilbert),
                pterm.transform,
            ))
        new.terms.append(copy_term)
    return new


###############################################################################
###############################################################################


_data.to.add_conversions(
    [
        (CuOperator, _data.Dense, CuOperator_from_Dense),
        # (CuOperator, CuPyDense, CuOperator_from_CuDense),
        (CuOperator, _data.Dia, CuOperator_from_Dia),
        (_data.Dense, CuOperator, Dense_from_CuOperator, 1e10),
    ]
)
_data.to.register_aliases(["densitymat_OperatorTerm", "CuOperator"], CuOperator)

_data.adjoint.add_specialisations([
    (CuOperator, CuOperator, CuOperator.adjoint),
])

_data.transpose.add_specialisations([
    (CuOperator, CuOperator, CuOperator.transpose),
])

_data.conj.add_specialisations([
    (CuOperator, CuOperator, CuOperator.conj),
])

_data.mul.add_specialisations([
    (CuOperator, CuOperator, CuOperator.__mul__),
])

_data.neg.add_specialisations([
    (CuOperator, CuOperator, CuOperator.__neg__),
])

_data.matmul.add_specialisations([(
    CuOperator, CuOperator, CuOperator,
    lambda left, right, scale=1.: left @ right * scale
)])

_data.add.add_specialisations([(
    CuOperator, CuOperator, CuOperator,
    lambda left, right, scale=1.: left + right * scale
)])

_data.sub.add_specialisations([
    (CuOperator, CuOperator, CuOperator, CuOperator.__sub__),
])

_data.diag.add_specialisations([
    (CuOperator, diags_CuOperator),
])

_data.identity.add_specialisations([
    (CuOperator, identity_CuOperator),
])

_data.zeros.add_specialisations([
    (CuOperator, zeros_CuOperator),
])

_data.kron.add_specialisations([
    (CuOperator, CuOperator, CuOperator, kron_CuOperator),
])

_data.permute.dimensions.add_specialisations([
    (CuOperator, CuOperator, dimensions_CuOperator),
])
