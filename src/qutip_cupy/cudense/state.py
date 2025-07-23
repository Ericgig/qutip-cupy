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

from qutip.core.data import Data, Dense
from qutip.core import data as _data
from qutip import settings

from ..dense import CuPyDense
from .utils import *


class CuState(Data):
    def __init__(self, arg, hilbert_dims=None, shape=None, copy=True):
        if isinstance(arg, (DensePureState, DenseMixedState)):
            if shape is not None: 
                pass
            elif isinstance(arg, DensePureState):
                shape = (np.prod(arg.hilbert_space_dims), 1)
            elif isinstance(arg, DenseMixedState):
                shape = (np.prod(arg.hilbert_space_dims),) * 2
            if hilbert_dims is None:
                hilbert_dims = arg.hilbert_space_dims
            base = arg

        elif isinstance(arg, CuPyDense):
            if shape is None: shape=arg.shape
            if hilbert_dims is None:
                hilbert_dims = arg.shape[:1]


            
            if arg.shape[0] != np.prod(hilbert_dims) or arg.shape[1] != 1:
                # TODO: Add sanity check for hilbert_dims
                base = DenseMixedState(settings.cuDensity["ctx"], hilbert_dims, 1, "complex128")
                sizes, offsets = base.local_info
                sls = tuple(slice(s, s+n) for s, n in zip(offsets, sizes))[:1]
                N = np.prod(sizes)
                if len(arg._cp) == N:
                    base.attach_storage(cp.array(arg._cp.reshape(hilbert_dims * 2)[sls].ravel(order="F"), copy=copy))
                else:
                    base.allocate_storage()
                    base.storage[:N] = arg._cp.reshape(hilbert_dims * 2)[sls].ravel(order="F")
                    
            else:
                base = DensePureState(settings.cuDensity["ctx"], hilbert_dims, 1, "complex128")
                sizes, offsets = base.local_info
                sls = tuple(slice(s, s+n) for s, n in zip(offsets, sizes))[:1]
                N = np.prod(sizes)
                if len(arg._cp) == N:
                    base.attach_storage(cp.array(arg._cp.reshape(hilbert_dims)[sls].ravel(order="F"), copy=copy))
                else:
                    base.allocate_storage()
                    base.storage[:N] = arg._cp.reshape(hilbert_dims)[sls].ravel(order="F")

        elif isinstance(arg, Data):
            arg = _data.to(_data.Dense, arg)
            if shape is None: shape=arg.shape
            if hilbert_dims is None:
                hilbert_dims = arg.shape[:1]

            if arg.shape[0] != np.prod(hilbert_dims) or arg.shape[1] != 1:
                # TODO: Add sanity check for hilbert_dims
                base = DenseMixedState(settings.cuDensity["ctx"], hilbert_dims, 1, "complex128")
                sizes, offsets = base.local_info
                sls = tuple(slice(s, s+n) for s, n in zip(offsets, sizes))[:-1]
                N = np.prod(sizes)
                print(base.local_info)
                print(sls)
                print(base.storage_size, np.prod(sizes))
                arr_np = arg.to_array().reshape(hilbert_dims * 2)[sls].ravel("F")
                base.allocate_storage()
                base.storage[:N] = cp.array(arr_np)
            else:
                base = DensePureState(settings.cuDensity["ctx"], hilbert_dims, 1, "complex128")
                sizes, offsets = base.local_info
                sls = tuple(slice(s, s+n) for s, n in zip(offsets, sizes))[:1]
                N = np.prod(sizes)
                arr_np = arg.to_array().reshape(hilbert_dims)[sls].ravel("F")
                base.allocate_storage()
                base.storage[:N] = cp.array(arr_np)

        else:
            raise NotImplementedError(type(arg))

        self.base = base
        self.transform = Transform.DIRECT
        super().__init__(shape=shape)

    def copy(self):
        return CuState(
            self.base.clone(cp.array(self.base.storage, copy=True)),
            shape=self.shape
        )

    def to_array(self, as_tensor=False):
        return self.to_cupy(as_tensor).get()

    def to_cupy(self, as_tensor=False):
        # TODO: Would this work with mpi?
        tensor = self.base.view()[..., 0]
        if not as_tensor:
            tensor = tensor.reshape(*self.shape, order="C")
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
        raise NotImplementedError()
        new = self.copy()
        new.transform = conj_transform[new.transform]
        return new

    def transpose(self):
        raise NotImplementedError()
        new = self.copy()
        new.transform = trans_transform[new.transform]
        return new

    def adjoint(self):
        raise NotImplementedError()
        new = self.copy()
        new.transform = adjoint_transform[new.transform]
        return new


def CuState_from_Dense(mat):
    return CuState(mat)


def CuState_from_CuPyDense(mat):
    return CuState(mat)


def Dense_from_CuState(mat):
    # TODO: does view work with MPI
    return _data.Dense(mat.to_array())


def CuPyDense_from_CuState(mat):
    # TODO: does view work with MPI
    return _data.CuPyDense(mat.to_cupy())


_data.to.add_conversions(
    [
        (CuState, _data.Dense, CuState_from_Dense),
        (CuState, CuPyDense, CuState_from_CuPyDense),
        (_data.Dense, CuState, Dense_from_CuState),
        (CuPyDense, CuState, CuPyDense_from_CuState),
    ]
)

_data.to.register_aliases(["CuState"], CuState)


@_data.trace.register(CuState)
def trace_cuState(mat):
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(...)

    return mat.base.trace()


@_data.inner.register(CuState, CuState)
def inner_cuState(left, right, scalar_is_ket=False):
    inner = left.base.inner_product(right.base)
    if self.shape == (1, 1) and not scalar_is_ket:
        inner = left.storage[0] * right.storage[0]
    else:
        inner = left.inner_product(right)
    return inner


@_data.kron.register(CuState, CuState, CuState)
def kron_cuState(left, right):
    if type(left.base) != type(right.base):
        raise TypeError(...)
    state = type(left.base)(
        settings.cuDensity["ctx"], 
        left.base.hilbert_space_dims + right.base.hilbert_space_dims, 
        1, 
        "complex128"
    )
    # right <--> left reversed since F ordered.
    kron = cp.kron(right.to_cupy(), left.to_cupy()).ravel(order="F")
    state.attach_storage(kron.copy())
    return CuState(state, copy=False)


@_data.mul.register(CuState, CuState)
def mul_cuState(mat, val):
    return mat * val


@_data.add.register(CuState, CuState, CuState)
def add_cuState(left, right, scale=1.):
    out = left.copy()
    out.base.inplace_accumulate(right.base, factor)
    return out


@_data.imul.register(CuState, CuState)
def imul_cuState(mat, val):
    mat.base.inplace_scale(val)
    return mat


@_data.iadd.register(CuState, CuState, CuState)
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

@_data.reshape.register(CuState, CuState)
def reshape_stack(matrix):
    raise NotImplementedError(...)
    return matrix


@_data.column_stack.register(CuState, CuState)
def column_stack(matrix):
    return CuState(matrix.base, shape=(matrix.shape[0] * matrix.shape[1], 1))

@_data.column_unstack.register(CuState, CuState)
def column_unstack(matrix, rows):
    return CuState(matrix.base, shape=(matrix.shape[0] / rows, rows))


def zeros_like_cuState(state):
    return CuState(state.base.clone(cp.zeros_like(state.base.storage, order="F")))
