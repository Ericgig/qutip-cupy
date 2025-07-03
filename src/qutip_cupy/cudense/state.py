try:
    import cuquantum.densitymat as cudense
    DensePureState = cudense.DensePureState
    DenseMixedState = cudense.DenseMixedState

except ImportError:
    class _Missing:
        ...

    cudense = None
    DensePureState = _Missing
    DenseMixedState = _Missing


from enum import Enum
from qutip.core.data import Data
from qutip.core import data as _data

from ..dense import CuPyDense


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


class CuState(Data):
    def __init__(self, arg, shape=None):
        if not isinstance(arg, (DensePureState, DenseMixedState)):
            raise TypeError(...)

        self.base = arg
        self.transform = Transform.DIRECT
        shape = (int(np.prod(arg.hilbert_space_dims)), ) * 2
        super().__init__(shape=shape)

    def copy(self):
        return self.base.clone(self.base.storage)

    def to_array(self, as_tensor=False):
        return self.to_cupy(as_tensor)

    def to_cupy(self, as_tensor=False):
        # TODO: Would this work with mpi?
        tensor = self.base.view()[..., 0]
        if not as_tensor:
            tensor = tensor.reshape(*self.shape)
        return tensor

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        if not isinstance(other, CuState):
            if isinstance(other, Data):
                return _data.add(self, other)
            return NotImplemented

        new = self.copy()
        new.base.inplace_accumulate(other.base, 1.)
        return new

    def __sub__(self, other):
        if not isinstance(other, CuState):
            if isinstance(other, Data):
                return _data.sub(self, other)
            return NotImplemented

        new = self.copy()
        new.base.inplace_accumulate(other.base, -1.)
        return new

    def __mul__(self, other):
        new = self.copy()
        new.base.inplace_scale(other.base, other)
        return new

    def __div__(self, other):
        return self * (1 / other)

    def conj(self):
        new = self.copy()
        new.transform = conj_transform[new.transform]
        return new

    def transpose(self):
        new = self.copy()
        new.transform = trans_transform[new.transform]
        return new

    def adjoint(self):
        new = self.copy()
        new.transform = adjoint_transform[new.transform]
        return new


def CuState_from_Dense(mat):
    ...
    return CuState(mat)


def CuState_from_CuDense(mat):
    ...
    return CuState(mat)


def Dense_from_CuState(mat):
    print("converting to Dense")
    return _data.Dense(mat.to_array())


def trace_cuState(mat):
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(...)

    return mat.base.trace()


def norm_cuState(mat):
    return mat.base.norm()


def inner_cuState(left, right, scalar_is_ket=False):
    inner = left.inner_product(right)
    if self.shape = (1, 1) and not scalar_is_ket:
        inner = left.storage[0] * right.storage[0]
    else:
        inner = left.inner_product(right)
    return inner


def kron_cuState(left, right):
    ...


def project_cuState(ket):
    ...


def matmul_outer(ket, bra):
    ...


def identity_cuState(N):
    ...


def one_element_cuState(shape, loc):
    ...


def zeros_cuState(shape):
    ...


def zeros_like_cuState(state):
    return CuState(state.base.clone(cp.zeros_like(state.base.storage, order="F")))
