
import copy


try:
    import cuquantum.densitymat as cudense
else:
    cudense = None

from qutip.core import data


def _transpose_Elementary(elem):
    if isinstance(oper, cudense.MultidiagonalOperator):
        new_oper = cudense.MultidiagonalOperator(oper.data, [-i for i in oper.offsets])
    elif isinstance(oper, cudense.DenseOperator):
        new_oper = cudense.DenseOperator(oper.data.T)
    else:
        raise NotImplementedError


def add_guess_hilbert(arg):
    hilbert = {}
    for terms, modes, duals in zip(self.base.terms, self.base.modes, self.base.duals):
        for term, mode, dual in zip(terms, modes, duals):
            N = term.shape[0]
            if dual:
                mode = -mode
            if mode in hilbert:
                assert hilbert[mode] == N
            else:
                hilbert[mode] = N
                if -mode in hilbert:
                    assert hilbert[-mode] == N

    N_hilbert = max(abs(k) for k in hilbert.keys())
    has_dual = min(k for k in hilbert.keys()) < 0
    all_dual = max(k for k in hilbert.keys()) < 0

    hilbert_list = [None] * N_hilbert
    for k, N in hilbert.items():
        hilbert_list[k] = N

    arg._hilbert_space_dims = tuple(hilbert_list)


def dual_to_extended(arg):
    hilbert = arg._hilbert_space_dims
    N_hilbert = len(hilbert)

    new_terms = []
    new_duals = []
    new_modes = []

    for terms, modes, duals in zip(arg.terms, arg.modes, arg.duals):
        for term, mode, dual in zip(terms, modes, duals):
            new_mode = [
                (i if dual else i + N_hilbert)
                for i in mode
            ]
            new_term = _transpose_Elementary(term) if dual else term)
        new_terms.append(new_term)
        new_duals.append(False)
        new_modes.append(new_mode)

    new = cudense.OperatorTerm()
    new.terms = new_terms
    new.modes = new_modes
    new.duals = new_duals
    new._coefficients = arg._coefficients
    new._hilbert_space_dims = hilbert + hilbert
    return new


def extended_to_dual(arg):
    hilbert = arg._hilbert_space_dims
    N_hilbert = len(hilbert) // 2

    new_terms = []
    new_duals = []
    new_modes = []

    for terms, modes, duals in zip(arg.terms, arg.modes, arg.duals):
        for term, mode, dual in zip(terms, modes, duals):
            new_mode = [
                (i if i < N_hilbert else i - N_hilbert)
                for i in mode
            ]
            new_term = _transpose_Elementary(term) if dual else term)
        new_terms.append(new_term)
        new_duals.append(mode[0] < N_hilbert)
        new_modes.append(new_mode)

    new = cudense.OperatorTerm()
    new.terms = new_terms
    new.modes = new_modes
    new.duals = new_duals
    new._coefficients = arg._coefficients
    new._hilbert_space_dims = hilbert[:N_hilbert]
    return new


def has_dual(arg):
    for duals in zip(self.base.duals):
        for dual in zip(duals):
            if dual:
                return True
    return False


class CuOperator(data.Data):
    def __init__(self, arg, shape=None, copy=True):
        if isinstance(
            arg,
            (cudense.MultidiagonalOperator, cudense.DenseOperator)
        ):
            oper_shape = arg.shape
            self.base = cudense.OperatorTerm()
            self.base.append_elementary_product([arg], [[0]], [[False]])
            self._type = "OperatorTerm"
            self.base._hilbert_space_dims = (oper_shape[0], )

        elif isinstance(arg, cudense.OperatorTerm):
            if arg.hilbert_space_dims is None:
                add_guess_hilbert(arg)
            if has_dual(arg):
                arg = dual_to_extended(arg)
            N = np.prod(arg.hilbert_space_dims)
            oper_shape = (N, N)
            self.base = arg
            self._type = "OperatorTerm"
        else:
            raise TypeError()

        if shape and shape != oper_shape:
            raise ValueError()

        super().__init__(oper_shape)

    def _match_hilbert(self, other):
        # TODO:
        # Add support for split
        # Add support for -1
        this = self.base._hilbert_space_dims
        other = other.base._hilbert_space_dims
        return this == other

    def copy(self):
        return CuOperator(self.base)

    def to_array(self):
        hilbert = self.base._hilbert_space_dims
        N = np.prod(hilbert)
        out = np.zeros((N, N), dtype=complex)
        for *term, coeff in zip(self.base.terms, self.base.modes, self.base.duals, self.base._coefficients):
            if not isinstance(coeff, cudense._internal.callbacks.ScalarCallbackCoefficient):
                raise NotImplementedError()
            termmat = np.eye(N, dtype=complex) * coeff._static_coeff
            for oper, mode, dual in zip(*term):
                mat = oper.to_array()
                try:
                    mat = mat.get()
                except:
                    pass
                mat = qt.data.Dense(mat.reshape(oper.shape + (-1,))[:, :, 0])
                if len(dual) != 1 or dual[0]:
                    raise NotImplementedError()
                idxs = list(range(len(hilbert)))
                sizes = []
                for i in mode:
                    sizes.append(hilbert[idxs.pop(i)])
                for i in idxs:
                    N = hilbert[i]
                    mat = qt.data.kron(mat, qt.data.identity(N))
                    sizes.append(N)
                mat = qt.data.permute.dimensions(mat, sizes, mode + idxs)
                termmat = mat.to_array() @ termmat
            out += termmat
        return out

    def conj(self):
        terms = []
        for term in self.base.terms:
            term_ops = []
            for oper in term:
                if isinstance(oper, cudense.MultidiagonalOperator):
                    new_oper = cudense.MultidiagonalOperator(oper.data.conj(), oper.offsets)
                elif isinstance(oper, cudense.DenseOperator):
                    new_oper = cudense.DenseOperator(oper.data.conj())
                else:
                    raise NotImplementedError

                term_ops.append(new_oper)
            terms.append(term_ops)
        new = cudense.OperatorTerm()
        new.terms = terms
        new.modes = self.base.modes
        new.duals = self.base.duals
        new._coefficients = self.base._coefficients
        new._hilbert_space_dims = self.base._hilbert_space_dims
        return CuOperator(new)

    def transpose(self):
        terms = []
        for term in self.base.terms:
            term_ops = []
            for oper in term:
                term_ops.append(_transpose_Elementary(oper))
            terms.append(term_ops[::-1])
        new = cudense.OperatorTerm()
        new.terms = terms
        new.modes = self.base.modes
        new.duals = self.base.duals
        new._coefficients = self.base._coefficients
        new._hilbert_space_dims = self.base._hilbert_space_dims
        return CuOperator(new)

    def adjoint(self):
        new = self.base.dag()
        new._hilbert_space_dims = self.base._hilbert_space_dims
        return CuOperator(new)

    def __add__(self, other):
        if self.base._hilbert_space_dims != other.base._hilbert_space_dims:
            raise ValueError()
        new = self.base + other.base
        new._hilbert_space_dims = self.base._hilbert_space_dims
        return CuOperator(new)

    def __sub__(self, other):
        if self.base._hilbert_space_dims != other.base._hilbert_space_dims:
            raise ValueError()
        new = self.base - other.base
        new._hilbert_space_dims = self.base._hilbert_space_dims
        return CuOperator(new)

    def __mul__(self, other):
        new = self.base * other
        new._hilbert_space_dims = self.base._hilbert_space_dims
        return CuOperator(new)

    def __matmul__(self, other):
        if self.base._hilbert_space_dims != other.base._hilbert_space_dims:
            raise ValueError()
        new = self.base * other.base
        new._hilbert_space_dims = self.base._hilbert_space_dims
        return CuOperator(new)

    def trace(self):
        return np.trace(self.to_array())

    def __neg__(self):
        new = self.base * -1
        new._hilbert_space_dims = self.base._hilbert_space_dims
        return CuOperator(new)

    def __div__(self, other):
        new = self.base * 1/other
        new._hilbert_space_dims = self.base._hilbert_space_dims
        return CuOperator(new)


def CuOperator_from_Dia(mat):
    dia_matrix = dia.as_scipy()
    offsets = list(dia_matrix.offsets)
    data = np.zeros((dia_matrix.shape[0], len(offsets)), dtype=complex)
    for i, offset in enumerate(offsets):
        end = None if offset == 0 else -abs(offset)
        data[:end, i] = dia_matrix.diagonal(offset)
    dia_op = MultidiagonalOperator(data, offsets)
    return CuOperator(dia_op)


def CuOperator_from_Dense(mat):
    return CuOperator(DenseOperator(mat.to_array()))


def CuOperator_from_CuDense(mat):
    return CuOperator(DenseOperator(mat._cp))


def Dense_from_CuOperator(mat):
    return Dense(mat.to_array())


def identity(dimension, scale=1):
    base = cudense.tensor_product(dtype="complex128")
    if scale != 1:
        base = base * scale
    base._hilbert_space_dims = (dimension, )
    return CuOperator(base)


def zeros(rows, cols):
    if rows != cols:
        raise NotImplementedError
    return identity(rows, 0)


def diags(diagonals, offsets=None, shape=None):
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
    left = left.base
    right = copy.copy(right.base)
    N = len(left._hilbert_space_dims)

    modes = []
    for mode in self.base.modes:
        shifted_mode = tuple(i + N for i in mode)
        modes.append(shifted_mode)
    right.modes = modes
    new = left * right
    new._hilbert_space_dims = left._hilbert_space_dims + right._hilbert_space_dims

    return CuOperator(new)


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
    assert matrix.base._hilbert_space_dims == hilbert
    new = deepcopy(matrix.base)
    permutation = list(order)
    sorted(permutation)
    new_hilbert = (matrix.base._hilbert_space_dims[i] for i in permutation)

    new_modes = []
    for modes in self.base.modes:
        new_mode = []
        for mode in modes:
            new_mode.append(order[mode])
        new_modes.append(new_mode)

    new.modes = new_modes
    new._hilbert_space_dims = new_hilbert

    return CuOperator(new)


from qutip.core import data
from .dense import CuPyDense

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
