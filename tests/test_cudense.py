import numpy as np
import pytest
cudense = pytest.importorskip("cuquantum.densitymat")

import qutip
import qutip_cupy
from qutip_cupy import CuState, CuOperator
import qutip.core.data as _data
import qutip.tests.core.data.test_mathematics as test_tools
from qutip.tests.core.data import conftest


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


def cases_cuoperator(hilbert):
    """Generate a random `CuPyDense` matrix with the given shape."""
    def factory(N_oper, mix=False):
        # TODO, add single operator acting on multiple mode.
        out = CuOperator(hilbert_dims=hilbert)
        for N in N_oper:
            part = 1.
            for _ in range(N):
                mode = np.random.randint(len(hilbert))
                size = abs(hilbert[mode])
                if np.random.randint(2):
                    part = conftest.random_dense((size, size), np.random.randint(2))
                else:
                    part = conftest.random_diag((size, size), 0.4, False)
                part = part * CuOperator(part, mode=(mode,), hilbert_dims=hilbert)
            out = out + part

    return [
        pytest.param(factory([]), id="zero"),
        pytest.param(factory([1]), id="simple"),
        pytest.param(factory([3]), id="m_prods"),
        pytest.param(factory([1, 1, 1]), id="m_terms"),
        pytest.param(factory([2, 3, 4], True), id="complex"),
    ]


test_tools._ALL_CASES = {
    CuState: lambda hilbert: [
        lambda: random_pure_custate(hilbert),
        lambda: random_mixed_custate(hilbert),
    ],
    CuOperator: cases_cuoperator,
}
test_tools._RANDOM = {
    CuState: lambda shape: [],
    CuOperator: lambda shape: [],
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
