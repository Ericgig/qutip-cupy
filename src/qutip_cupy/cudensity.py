
import copy


try:
    import cuquantum.densitymat as cudense
else:
    cudense = None

from qutip.core import data


class CuOperator(data.Data):
    def __init__(self, arg, shape=None, copy=True):
        if isinstance(arg, _CTEMAT):
            oper_shape = (arg.shape, arg.shape)
            self.base = arg
            self._type = "cte"

        elif isinstance(
            arg,
            (cudense.MultidiagonalOperator, cudense.DenseOperator)
        ):
            oper_shape = arg.shape
            self.base = cudense.OperatorTerm()
            self.base.append_elementary_product([arg], [[0]], [[False]])
            self._type = "OperatorTerm"
            self.base._hilbert_space_dims = (oper_shape[0], )

        elif isinstance(arg, cudense.OperatorTerm):
            N = np.prod(arg.hilbert_space_dims)
            oper_shape = (N, N)
            self.base = arg
            self._type = "OperatorTerm"

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
                if isinstance(oper, cudense.MultidiagonalOperator):
                    new_oper = cudense.MultidiagonalOperator(oper.data, [-i for i in oper.offsets])
                elif isinstance(oper, cudense.DenseOperator):
                    new_oper = cudense.DenseOperator(oper.data.T)
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


def qobj2dense(qobj, dtype=np.complex128):
    return DenseOperator(qobj.full().astype(dtype))


def qobj2multidiagonal(qobj, dtype=np.complex128):
    dia_matrix = qobj.to("dia").data.as_scipy()
    offsets = list(dia_matrix.offsets)
    data = np.zeros((dia_matrix.shape[0], len(offsets)), dtype=dtype)
    for i, offset in enumerate(offsets):
        end = None if offset == 0 else -abs(offset)
        data[:end, i] = dia_matrix.diagonal(offset)
    dia_op = MultidiagonalOperator(data, offsets)
    return dia_op


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
    return DenseOperator(mat.to_array())


def CuOperator_from_CuDense(mat):
    return DenseOperator(mat._cp)


def Dense_from_CuOperator(mat):
    return Dense(mat.to_array())


def kron_CuOperator_CuOperator(left, right):
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
