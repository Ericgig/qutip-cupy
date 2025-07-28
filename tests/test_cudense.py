import numpy as np
import cupy as cp
import pytest
cudense = pytest.importorskip("cuquantum.densitymat")

import qutip
import qutip_cupy
from qutip_cupy.cudense import CuState, CuOperator
from qutip_cupy.cudense.cudense import ProdTerm, Term
from qutip_cupy.cudense.utils import Transform

import qutip.core.data as _data
import qutip.tests.core.data.test_mathematics as test_tools
from qutip.tests.core.data.conftest import (
    random_csr, random_dense, random_diag
)


def random_pure_custate(hilbert):
    """Generate a random `CuPyDense` matrix with the given shape."""
    N = abs(np.prod(hilbert))
    out = (
        cp.random.rand(N, 1) + 1j * cp.random.rand(N, 1)
    ).astype(cp.complex128)
    out = qutip_cupy.CuPyDense._raw_cupy_constructor(out)
    return CuState(out, hilbert, copy=False)


def random_mixed_custate(hilbert):
    """Generate a random `CuPyDense` matrix with the given shape."""
    N = abs(np.prod(hilbert))
    out = (
        cp.random.rand(N, N) + 1j * cp.random.rand(N, N)
    ).astype(cp.complex128)
    out = qutip_cupy.CuPyDense._raw_cupy_constructor(out)
    return CuState(out, hilbert, copy=False)


def _rand_transform(gen):
    """
    Random transform between raw, dag, T, conj, with bias toward common cases.
    """
    return gen.choice(list(Transform), p=[0.4, 0.15, 0.15, 0.3])


def _rand_elementary_oper(size, gen):
    if gen.uniform() < 0.5:
        mat = random_diag((size, size), gen.uniform()*0.4, False, gen)
    elif gen.uniform() < 0.8:
        mat = random_dense((size, size), gen.uniform() > 0.5, gen)
    else
        mat = random_csr((size, size), gen.uniform()*0.4, False, gen)

    if gen.uniform() < 0.5:
        array_type = np if gen.uniform() < 0.5 else cp
        if isinstance(mat, _data.Dia):
            dia_matrix = oper.as_scipy()
            offsets = list(dia_matrix.offsets)
            data = array_type.zeros(
                (dia_matrix.shape[0], len(offsets)),
                dtype=complex,
            )
            for i, offset in enumerate(offsets):
                end = None if offset == 0 else -abs(offset)
                data[:end, i] = dia_matrix.diagonal(offset)
            mat = cudense.MultidiagonalOperator(data, offsets)

        else:
            mat = cudense.DenseOperator(array_type.array(oper.to_array()))

    return mat


def random_CuOperator(hilbert_dims, N_elementary, seed):
    """
    Generate a random `CuOperator` matrix with the given hilbert_dims.
    """
    generator = np.random.default_rng(seed)
    out = CuOperator(hilbert_dims=hilbert)
    for N in N_elementary:
        term = Term([], generator.normal() + 1j * generator.normal())
        for _ in range(N):
            mode = np.random.randint(len(hilbert))
            size = abs(hilbert[mode])
            oper = _rand_elementary_oper(size, gen)

            term.prod_terms.append(ProdTerm(oper, mode, _rand_transform(gen)))
        out.terms.append(term)
    return out


def cases_cuoperator(hilbert):
    """Generate a random `CuPyDense` matrix with the given shape."""

    def factory(N_elementary, seed):
        return lambda: random_CuOperator(hilbert, N_elementary, seed)

    cases = []

    cases.append(pytest.param(factory([], 0), id="zero"))
    cases.append(pytest.param(factory([0], 0), id="id"))
    seed = random.randint(0, 2**31)
    cases.append(pytest.param(factory([1], seed), id=f"simple_{seed}"))
    seed = random.randint(0, 2**31)
    cases.append(pytest.param(factory([3], seed), id=f"3_prods_{seed}"))
    seed = random.randint(0, 2**31)
    cases.append(pytest.param(factory([1, 1, 1], seed), id=f"3_terms_{seed}"))
    seed = random.randint(0, 2**31)
    cases.append(pytest.param(factory([1, 2, 3], seed), id=f"complex_{seed}"))

    return cases


test_tools._ALL_CASES = {
    CuState: lambda hilbert: [
        lambda: random_pure_custate(hilbert),
        lambda: random_mixed_custate(hilbert),
    ],
    CuOperator: cases_cuoperator,
}

test_tools._RANDOM = {
    CuState: lambda shape: [],
    CuOperator: lambda hilbert: [lambda: random_CuOperator(hilbert, [2], 0)],
}

_unary_hilbert = [
    (2,),
    (3,),
    (-6,),
    (2, 3),
    (-2, 2, 2),
    (2, 3, -4),
]


_compatible_hilbert = [
    ((2, ), (2, )),
    ((2, 3), (2, 3)),
    ((2, 3), (-6, )),
    ((2, 3, -4), (-6, 2, 2)),
    ((2, -4), (-4, 2)),
]


_imcompatible_hilbert = [
    ((2, ), (3, )),
    ((2, 3), (6)),
    ((2, 3), (3, 2)),
    ((2, 3, -4), (6, 2, 2)),
    ((-2, -4), (4, -2)),
]


class TestAdd(test_tools.TestAdd):
    specialisations = [
        pytest.param(lambda x, y, s; x + y * s, CuOperator, CuOperator, CuOperator),
    ]

    shapes = _compatible_hilbert
    bad_shapes = _imcompatible_hilbert


class TestSub(test_tools.TestSub):
    specialisations = [
        pytest.param(lambda x, y; x - y, CuOperator, CuOperator, CuOperator),
    ]

    shapes = _compatible_hilbert
    bad_shapes = _imcompatible_hilbert


class TestMatmul(test_tools.TestMatmul):
    specialisations = [
        pytest.param(lambda x, y; x @ y, CuOperator, CuOperator, CuOperator),
    ]

    shapes = _compatible_hilbert
    bad_shapes = _imcompatible_hilbert


class TestMul(test_tools.TestMul):
    specialisations = [
        pytest.param(lambda op, scale: op * scale, CuOperator, CuOperator),
    ]

    shapes = _unary_hilbert
    bad_shapes = []


class TestNeg(test_tools.TestNeg):
    specialisations = [
        pytest.param(lambda op: -op, CuOperator, CuOperator),
    ]

    shapes = _unary_hilbert
    bad_shapes = []


class TestAdjoint(test_tools.TestAdjoint):
    specialisations = [
        pytest.param(lambda op: op.adjoint(), CuOperator, CuOperator),
    ]

    shapes = _unary_hilbert
    bad_shapes = []


class TestConj(test_tools.TestConj):
    specialisations = [
        pytest.param(lambda op: op.conj(), CuOperator, CuOperator),
    ]

    shapes = _unary_hilbert
    bad_shapes = []


class TestTranspose(test_tools.TestTranspose):
    specialisations = [
        pytest.param(lambda op: op.transpose(), CuOperator, CuOperator),
    ]

    shapes = _unary_hilbert
    bad_shapes = []


class TestKron(test_tools.TestKron):
    specialisations = [
        pytest.param(_data.kron, CuOperator, CuOperator, CuOperator),
    ]

    shapes = _compatible_hilbert + _imcompatible_hilbert
    bad_shapes = []
