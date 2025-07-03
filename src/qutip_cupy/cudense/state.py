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

import numpy as np
import cupy as cp

from qutip.core.data import Data
from qutip.core import data as _data

from ..dense import CuPyDense
from .utils import *


class CuState(Data):
    def __init__(self, arg, hilbert_dims=None, shape=None, copy=True):
        if isinstance(arg, (DensePureState, DenseMixedState)):
            base = arg
        elif isinstance(arg, CuPyDense):
            if arg.shape[0] == arg.shape[1]:
                # TODO: Add sanity check for hilbert_dims
                base = DenseMixedState(settings.cuDensity["ctx"], hilbert_dims, 1, "complex128")
                base.allocate_storage(cp.array(arg.arg, copy=copy).ravel(order="F"))
            else:
                base = DensePureState(settings.cuDensity["ctx"], hilbert_dims, 1, "complex128")
                base.allocate_storage(cp.array(arg.arg, copy=copy).ravel(order="F"))

        self.base = base
        self.transform = Transform.DIRECT
        # TODO: Add sanity check for shape
        shape = (int(np.prod(base.hilbert_space_dims)), ) * 2
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
        new.base.inplace_scale(other)
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


def inner_cuState(left, right, scalar_is_ket=False):
    inner = left.inner_product(right)
    if self.shape = (1, 1) and not scalar_is_ket:
        inner = left.storage[0] * right.storage[0]
    else:
        inner = left.inner_product(right)
    return inner


def kron_cuState(left, right):
    ...


@_data.imul.register(CuState, CuState)
def imul_cuState(mat, val):
    return mat.base.inplace_scale(val)


def iadd_cuState(left, right, factor=1.):
    left.base.inplace_accumulate(right.base, factor)
    return left


@_data.norm.frobenius.register(CuState)
def frobenius_cuState(mat):
    return mat.base.norm()


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
