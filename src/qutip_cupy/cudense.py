try:
    import cuquantum.densitymat as cudense
else:
    cudense = None

from qutip.core import data as _data
from collections import namedtuple
from enum import Enum
from typing import NamedTuple, Any
from dataclasses import dataclass
from qutip.core.data import Data
import qutip.data as _data




def _transpose_Elementary(elem):
    if isinstance(oper, cudense.MultidiagonalOperator):
        new_oper = cudense.MultidiagonalOperator(oper.data, [-i for i in oper.offsets])
    elif isinstance(oper, cudense.DenseOperator):
        new_oper = cudense.DenseOperator(oper.data.T)
    else:
        raise NotImplementedError


###############################################################################
###############################################################################


class obj:
    ...

cudense = obj()
cudense.MultidiagonalOperator = None
cudense.DenseOperator = None
cudense.OperatorTerm = None
cudense._internal = obj()
cudense._internal.callbacks = obj()
cudense._internal.callbacks.ScalarCallbackCoefficient = None

Term = namedtuple("Term", ["prodterms", "factor"])
ProdTerm = namedtuple("ProdTerm", ["operators", "hilbert", "transform"])


class Transform(Enum):
    DIRECT = 0
    CONJ = 1
    TRANSPOSE = 2
    ADJOINT = 3


conj = {
    Transform.DIRECT : Transform.CONJ,
    Transform.CONJ : Transform.DIRECT,
    Transform.TRANSPOSE : Transform.ADJOINT,
    Transform.ADJOINT : Transform.TRANSPOSE,
}

trans = {
    Transform.DIRECT : Transform.TRANSPOSE,
    Transform.CONJ : Transform.ADJOINT,
    Transform.TRANSPOSE : Transform.DIRECT,
    Transform.ADJOINT : Transform.CONJ,
}

adjoint = {
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


def guess_hilbert(arg):
    hilbert = {}
    for terms, modes in zip(self.base.terms, self.base.modes):
        for term, mode in zip(terms, modes):
            for i, M in enumerate(mode):
                if m in hilbert:
                    assert hilbert[M] == term.shape[i]
                else:
                    hilbert[M] = term.shape[i]

    N_hilbert = max(abs(k) for k in hilbert.keys())

    hilbert_list = [None] * N_hilbert
    for k, N in hilbert.items():
        hilbert_list[k] = N
    hilbert_list.append(...)

    return tuple(hilbert_list)


def _has_dual(arg):
    for duals in zip(self.base.duals):
        for dual in zip(duals):
            if dual:
                return True
    return False


class CuOperator(Data):
    terms: list
    hilbert_dims: tuple

    def __init__(self, arg, shape=None, copy=True):
        self.terms = []
        self.hilbert_dims = ()
        oper_shape = None

        if arg is None:
            self.hilbert_dims = (shape[0], )
            oper_shape = shape
        elif isinstance(
            arg,
            (cudense.MultidiagonalOperator, cudense.DenseOperator)
        ):
            oper_shape = arg.shape
            arg = arg.copy() if copy else arg
            self.terms.append(
                Term([ProdTerm(arg, (0,), Transform.DIRECT)], 1.+0j)
            )
            self.hilbert_dims = (arg.shape[0], )

        elif isinstance(arg, cudense.OperatorTerm):
            if arg.hilbert_space_dims is None:
                hilbert_dims = guess_hilbert(arg)
            else:
                hilbert_dims = arg.hilbert_space_dims

            if all(isinstance(n, int) for n in arg.hilbert_space_dims):
                N = np.prod(arg.hilbert_space_dims)
                oper_shape = (N, N)

            has_dual = _has_dual(arg)
            N = len(hilbert_dims)
            for terms, modes, duals, coeff in zip(arg.terms, arg.modes, arg.duals, arg._coefficients):
                if not isinstance(coeff, cudense._internal.callbacks.ScalarCallbackCoefficient):
                    raise NotImplementedError()
                terms = Term([], factor=coeff._static_coeff)

                for term, mode, dual in zip(terms, modes, duals):
                    term = term.copy() if copy else term
                    if has_dual and not dual:
                        mode = tuple(i + N for i in mode)
                    terms.prod_terms.append(ProdTerm(term, mode, Transform.DIRECT))

            if has_dual:
                self.hilbert_dims = hilbert_dims + hilbert_dims
            else:
                self.hilbert_dims = hilbert_dims

        elif isinstance(arg, Data) and not isinstance(arg, CuOperator):
            oper_shape = arg.shape
            arg = arg.copy() if copy else arg
            self.terms.append(
                Term([ProdTerm(arg, (0,), Transform.DIRECT)], 1.+0j)
            )
            self.hilbert_dims = (arg.shape[0], )

        else:
            raise TypeError()

        if shape and oper_shape and shape != oper_shape:
            raise ValueError()
        if not (shape or oper_shape):
            raise ValueError()
        if shape[0] != shape[1]:
            raise ValueError()

        super().__init__(oper_shape)


    def copy(self):
        new = CuOperator(shape=self.shape)
        new.hilbert_dims = self.hilbert_dims
        for term in self.terms:
            copy_term = Term([], factor=term.factor)
            for pterm in term.prod_terms:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator.copy(),
                    pterm.hilbert,
                    pterm.transform,
                ))
            new.terms.append(copy_term)
        return new

    def to_array(self):
        hilbert = self.hilbert_dims
        out = np.zeros(*self.shape, dtype=complex)

        for term in zip(self.terms):
            termmat = np.eye(N, dtype=complex) * term.factor
            for prod_term in term.prod_terms:
                mat = prod_term.operator.to_array()
                try:
                    # if cupy array, get numpy
                    mat = mat.get()
                except:
                    pass
                if len(mat.shape) > 2:
                    mat = mat.reshape(oper.shape + (-1,))[:, :, 0]

                mat = apply_transformation(mat, prod_term.transform)

                idxs = list(range(len(hilbert)))
                sizes = []
                for i in mode:
                    sizes.append(hilbert[idxs.pop(i)])
                for i in idxs:
                    N = hilbert[i]
                    mat = np.kron(mat, np.eye(N))
                    sizes.append(N)
                mat = _data.permute.dimensions(_data.Dense(mat), sizes, mode + idxs, copy=False).to_array()
                termmat = mat @ termmat
            out += termmat
        return out

    def conj(self):
        new = CuOperator(shape=self.shape)
        new.hilbert_dims = self.hilbert_dims
        for term in self.terms:
            copy_term = Term([], factor=term.factor)
            for pterm in term.prod_terms:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator.copy(),
                    pterm.hilbert,
                    conj[pterm.transform],
                ))
            new.terms.append(copy_term)
        return new

    def transpose(self):
        new = CuOperator(shape=self.shape)
        new.hilbert_dims = self.hilbert_dims
        for term in self.terms:
            copy_term = Term([], factor=term.factor)
            for pterm in term.prod_terms[::-1]:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator.copy(),
                    pterm.hilbert,
                    transpose[pterm.transform],
                ))
            new.terms.append(copy_term)
        return new

    def adjoint(self):
        new = CuOperator(shape=self.shape)
        new.hilbert_dims = self.hilbert_dims
        for term in self.terms:
            copy_term = Term([], factor=term.factor)
            for pterm in term.prod_terms[::-1]:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator.copy(),
                    pterm.hilbert,
                    adjoint[pterm.transform],
                ))
            new.terms.append(copy_term)
        return new

    def __neg__(self, other):
        if self.hilbert_dims != other.hilbert_dims:
            raise ValueError()
        return other * -1

    def __add__(self, other):
        if not isinstance(other, CuOperator):
            if isinstance(other, Data):
                return _data.add(self, other)
            return NotImplemented

        if self.hilbert_dims != other.hilbert_dims:
            raise ValueError()
        new = self.copy()
        other = other.copy()
        new += other.terms
        return new

    def __sub__(self, other):
        if not isinstance(other, CuOperator):
            if isinstance(other, Data):
                return _data.sub(self, other)
            return NotImplemented

        if self.hilbert_dims != other.hilbert_dims:
            raise ValueError()
        return self + -other

    def __mul__(self, other):
        new = CuOperator(shape=self.shape)
        new.hilbert_dims = self.hilbert_dims
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
        new = self * (1 / other)
        new._hilbert_space_dims = self.base._hilbert_space_dims
        return CuOperator(new)

    def __matmul__(self, other):
        if not isinstance(other, CuOperator):
            if isinstance(other, Data):
                return _data.matmul(self, other)
            return NotImplemented

        if self.hilbert_dims != other.hilbert_dims:
            raise ValueError()
        new = CuOperator(shape=self.shape)
        new.hilbert_dims = self.hilbert_dims
        left = self.copy()
        right = other.copy()

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
        for term in self.terms:
            cuterm = 1
            for pterm in term.prod_terms:
                cuterm = cudense.OperatorTerm((apply_transformation(pterm.operator, pterm.transform), pterm.hilbert)) * cuterm
            out += cuterm
        return out


def identity_CuOperator(dimension, scale=1):
    new = CuOperator(shape=(dimension, dimension))
    new.terms.append(Term([], complex(scale)))
    return new


def zeros_CuOperator(rows, cols):
    return CuOperator(shape=(rows, cols))


def diags_CuOperator(diagonals, offsets=None, shape=None):
    N = None
    if shape is not None:
        assert shape[0] == shape[1]
        N = shape[0]

    if N is None:
        N = len(diagonals[0]) + abs(offsets[0])

    data = np.zeros((N, len(offsets)), dtype=complex)
    for i in range(len(diagonals)):
        diag = diagonals[i]
        data[:len(diag), i] = np.array(diag)

    out = cudense.MultidiagonalOperator(data, offsets)
    return CuOperator(out)


def kron_CuOperator(left, right):
    N = len(left.hilbert_dims)
    S = left.shape[0] * right.shape[0]

    left_ext = CuOperator(shape=(S, S))
    left_ext.hilbert_dims = left_ext.hilbert_dims + left_ext.hilbert_dims
    for term in left.terms:
        copy_term = Term([], factor=term.factor)
        for pterm in term.prod_terms:
            copy_term.prod_terms.append(ProdTerm(
                pterm.operator.copy(),
                pterm.hilbert,
                pterm.transform,
            ))
        left_ext.terms.append(copy_term)

    right_shifted = CuOperator(shape=(S, S))
    right_shifted.hilbert_dims = left.hilbert_dims + right.hilbert_dims
    for term in self.terms:
        copy_term = Term([], factor=term.factor)
        for pterm in term.prod_terms:
            copy_term.prod_terms.append(ProdTerm(
                pterm.operator.copy(),
                (i + N for i in pterm.hilbert),
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
    assert matrix.hilbert_dims == hilbert
    new = CuOperator(shape=self.shape)
    permutation = np.argsort(order)
    new.hilbert_dims = (self.hilbert_dims[i] for i in permutation)

    for term in self.terms:
        copy_term = Term([], factor=term.factor)
        for pterm in term.prod_terms:
            copy_term.prod_terms.append(ProdTerm(
                pterm.operator.copy(),
                (order[mode] for mode in pterm.hilbert),
                pterm.transform,
            ))
        new.terms.append(copy_term)
    return new


###############################################################################
###############################################################################


data.to.add_conversions(
    [
        (CuOperator, data.Dense, CuOperator_from_Dense),
        (CuOperator, CuPyDense, CuOperator_from_CuDense),
        (CuOperator, data.Dia, CuOperator_from_Dia),
        (data.Dense, CuOperator, Dense_from_CuOperator),
    ]
)
data.to.register_aliases(["densitymat_OperatorTerm", "CuOperator"], CuOperator)

data.adjoint.add_specialisations([
    (CuOperator, CuOperator, CuOperator.adjoint),
])

data.transpose.add_specialisations([
    (CuOperator, CuOperator, CuOperator.transpose),
])

data.conj.add_specialisations([
    (CuOperator, CuOperator, CuOperator.conj),
])

data.trace.add_specialisations([
    (CuOperator, CuOperator, CuOperator.trace),
])

data.mul.add_specialisations([
    (CuOperator, CuOperator, CuOperator.__mul__),
])

data.neg.add_specialisations([
    (CuOperator, CuOperator, CuOperator.__neg__),
])

data.matmul.add_specialisations([
    (CuOperator, CuOperator, CuOperator.__matmul__),
])

data.add.add_specialisations([
    (CuOperator, CuOperator, CuOperator.__add__),
])

data.sub.add_specialisations([
    (CuOperator, CuOperator, CuOperator.__sub__),
])

data.diag.add_specialisations([
    (CuOperator, diags),
])

data.identity.add_specialisations([
    (CuOperator, identity),
])

data.zeros.add_specialisations([
    (CuOperator, zeros),
])

data.kron.add_specialisations([
    (CuOperator, CuOperator, CuOperator, kron_CuOperator),
])

data.permute.dimensions.add_specialisations([
    (CuOperator, CuOperator, dimensions_CuOperator),
])
