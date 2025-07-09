try:
    import cuquantum.densitymat as cudense
    MultidiagonalOperator = cudense.MultidiagonalOperator
    DenseOperator = cudense.DenseOperator
    OperatorTerm = cudense.OperatorTerm
    tensor_product = cudense.tensor_product
    ScalarCallbackCoefficient = cudense._internal.callbacks.ScalarCallbackCoefficient

except ImportError:
    class _Missing:
        ...

    cudense = None
    MultidiagonalOperator = _Missing
    DenseOperator = _Missing
    OperatorTerm = _Missing
    ScalarCallbackCoefficient = _Missing
    tensor_product = _Missing


from enum import Enum
from typing import NamedTuple, Any
import itertools
import numpy as np

from qutip.core.data import Data
from qutip.core import data as _data

from ..dense import CuPyDense
from .utils import *
# __all__ = ["CuOperator"]


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


def _oper_to_ElementaryOperator(oper, hilbert_idx, hilbert_dims, copy=False):
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
        dia_matrix = oper.as_scipy()
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
    """
    Pseudo symbolic data layer that follow a structure close to
    cuDensity OperatorTerm.

    Only support square operators.

    The operator is composed by a sum of terms, which reach term composed of
    multiple product:

    op = sum_i term[i].factor * prod(term[i].prod_terms)

    prod(term[i].prod_terms) =
        expand_operator(oper[N-1]^trans[N-1], hilbert[N-1], hilbert_dims) @
        ...
        expand_operator(oper[1]^trans[1], hilbert[1], hilbert_dims) @
        expand_operator(oper[0]^trans[0], hilbert[0], hilbert_dims)

    This object always has (partial) knowledge of the full hilbert space.
    There is no dual representation, super operator will have the hilbert space
    doubled, with the operation being applied to the right first and
    transposed.
    """
    terms: list
    hilbert_dims: tuple

    def __init__(self, arg=None, shape=None, copy=True, hilbert_dims=None):
        self.terms = []
        self.hilbert_dims = ()
        self._oper = None
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

    @property
    def hilbert_space_dims(self):
        return tuple(abs(i) for i in self.hilbert_dims)

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
        hilbert = self.hilbert_space_dims
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
                    sizes.append(hilbert[idxs.pop(i)])
                for i in idxs:
                    N = hilbert[i]
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
        other._update_hilbert(new.hilbert_dims)
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

    def to_OperatorTerm(self, dual=False, copy=True, hilbert_dims=None):
        # TODO: Dim input instead of dual?
        if hilbert_dims is not None:
            self = self.copy()
            if dual:
                hilbert_dims = hilbert_dims + hilbert_dims
            self._update_hilbert(hilbert_dims)
        out = OperatorTerm(dtype="complex128")
        if not dual:
            for term in self.terms:
                cuterm = 1.
                for pterm in term.prod_terms:
                    oper = _apply_transformation(pterm.operator, pterm.transform)
                    oper = _oper_to_ElementaryOperator(oper, pterm.hilbert, self.hilbert_space_dims, copy)
                    cuterm = tensor_product((oper, pterm.hilbert)) * cuterm
                out += (cuterm * term.factor)
        else:
            N_hilbert = len(self.hilbert_dims) // 2
            # TODO: make this tests weak compare?
            assert self.hilbert_dims[:N_hilbert] == self.hilbert_dims[N_hilbert:]
            for term in self.terms:
                cuterm = 1.
                for pterm in term.prod_terms:
                    if all(i < N_hilbert for i in pterm.hilbert):
                        oper = _apply_transformation(pterm.operator, pterm.transform)
                        oper = _oper_to_ElementaryOperator(oper, pterm.hilbert, self.hilbert_space_dims, copy)
                        cuterm = tensor_product((oper, pterm.hilbert, [True])) * cuterm

                    elif any(i < N_hilbert for i in pterm.hilbert):
                        raise NotImplementedError

                    else:
                        oper = _apply_transformation(pterm.operator, pterm.transform)
                        oper = _oper_to_ElementaryOperator(oper, pterm.hilbert, self.hilbert_space_dims, copy)
                        cuterm = tensor_product(
                            (oper, tuple(i - N_hilbert for i in pterm.hilbert))
                        ) * cuterm
                out += (cuterm * term.factor)
            
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


_data.to.add_conversions(
    [
        (CuOperator, _data.Dense, CuOperator_from_Dense),
        (CuOperator, CuPyDense, CuOperator_from_CuDense),
        (CuOperator, _data.Dia, CuOperator_from_Dia),
        (_data.Dense, CuOperator, Dense_from_CuOperator, 1e10),
    ]
)
_data.to.register_aliases(["densitymat_OperatorTerm", "CuOperator"], CuOperator)


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


def extract_CuOperator(mat, format=None, copy=True):
    # TODO: Better name, other input for dual?
    if format not in [None, "OperatorTerm", "DualOperatorTerm"]:
        raise ValueError(...)

    dual = format == "DualOperatorTerm"
    return mat.to_OperatorTerm(dual=dual, copy=copy)


def dimensions_CuOperator(matrix, hilbert, order):
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

_data.extract.add_specialisations([
    (CuOperator, extract_CuOperator),
])
