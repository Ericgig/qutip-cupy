from enum import Enum
from functools import partial
from cuquantum.densitymat import CPUCallback, DenseOperator

from qutip.core.tensor import _reverse_partial_tensor, tensor
from qutip.core.superoperator import spre, spost
from qutip.core.cy._element import _BaseElement, _FuncElement, _MapElement, _ProdElement, _EvoElement
from qutip.core.data import Dia
from qutip.core import coefficient


__all__ = [
    "Transform",
    "conj_transform",
    "trans_transform",
    "adjoint_transform",
    "_compare_hilbert"
]


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

def Oper_to_cupy(oper, ctx):
    dims = oper.hilbert_space_dims
    N = np.prod(dims)
    id_ = DensePureState(ctx, dims, N, "complex128")_coe
    id_.allocate_storage()
    id_.storage[::N+1] = 1
    out = DensePureState(ctx, dims, N, "complex128")
    out.allocate_storage()
    
    oper.prepare_action(ctx, id_)
    oper.compute_action(0., None, id_, out)
    return out.view()


def Operterm_to_cupy(term, hdims, ctx):
    oper = Operator(hdims, (term,))
    return Oper_to_cupy(oper, ctx)


def wrap_coeff(coeff):
    return CPUCallback(lambda t, _: coeff(t))


def _wrap_callable(func):
    sample = func(0)
        shape = sample._dims._get_tensor_shape()
        perm = sample._dims._get_tensor_perm()
        num_mode = len(shape) // 2
    
    if sample.dtype is Dia and num_mode == 1:
        dia_matrix = sample.as_scipy()
        offsets = list(dia_matrix.offsets)

        def wrapped(t, _=None):
            # TODO: Should we make this a class for pickling?
            dia_matrix = func(t).as_scipy()
            arr_shape = (dia_matrix.shape[0], len(offsets))
            data = np.zeros(arr_shape, dtype=complex)

            for i, offset in enumerate(offsets):
                end = None if offset == 0 else -abs(offset)
                data[:end, i] = dia_matrix.diagonal(offset)

            return data

        out = MultidiagonalOperator(wrapped(0), offsets, callback=CPUCallback(wrapped))

    else:
        if sample.dtype is Dia:
            print("Callable QobjEvo converted to dense!")
        
        def wrapped(t, _=None):
            # TODO: Should we make this a class for pickling?
            arr = func(t).full()
            arr = arr.reshape(*shape)
            return arr.transpose(*perm)

        out = DenseOperator(wrapped(0), CPUCallback(wrapped))

    return out, num_mode


def wrap_funcelement(element, args, dual, hilbert_dims, anti=False):
    if not isinstance(element, _BaseElement):
        element = _FuncElement(element, args)

    if isinstance(element, _FuncElement):
        oper, num_mode = _wrap_callable(element.qobj)
        out = tensor_product((oper, tuple(range(num_modes)) ), dtype="complex128")

    elif isinstance(element, _MapElement):
        oper, num_mode = _wrap_callable(element._base.qobj)
        as_qobj = Qobj( CuOperator(oper), dims=element._base.qobj(0)._dims )
        for transform in element.transform:
            as_qobj = transform(as_qobj)
        if as_qobj.dtype is not CuOperator:
            oper, num_mode = _wrap_callable(element.qobj)
            out = tensor_product((oper, tuple(range(num_modes)) ), dtype="complex128")
        else:
            coeff = conj(element._coeff) if anti else element._coeff
            out = as_qobj.data.to_OperatorTerm(dual, hilbert_dims=hilbert_dims) * coeff

    elif isinstance(element, _ProdElement):
        left = CuOperator(wrap_funcelement(element.left, None, dual, hilbert_dims, self._conj != anti))
        right = CuOperator(wrap_funcelement(element.left, None, dual, hilbert_dims, self._conj != anti))
        as_qobj = Qobj(left @ right)
        
        if as_qobj.dtype is not CuOperator:
            oper, num_mode = _wrap_callable(element.qobj)
            coeff = make_CPUcall( coefficient(element.coeff).conj() ) if anti else make_CPUcall(element.coeff)
            out = tensor_product((oper, tuple(range(num_modes)) ), dtype="complex128") * coeff
        else:
            out = as_qobj.data.to_OperatorTerm(dual, hilbert_dims=hilbert_dims)
        

    elif isinstance(element, _EvoElement):
        qobj = element._qobj
        coeff = make_CPUcall(element._coefficient.conj()) if anti else make_CPUcall(element._coefficient)
        out = qobj.data.to_OperatorTerm(dual, hilbert_dims=self.hilbert_space_dims) * coeff

    elif isinstance(element, _ConstantElement):
        qobj = element._qobj
        out = qobj.data.to_OperatorTerm(dual, hilbert_dims=self.hilbert_space_dims)

    else:
        raise NotImplementedError(type(element))

    return out
        
            
        


















