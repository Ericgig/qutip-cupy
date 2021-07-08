from qutip_cupy import CuPyDense
import numpy as np
import pytest

@pytest.fixture(scope="function", params=((1, 2), (5, 10), (7, 3), (2, 5)))
def shape(request):
    return request.param


class TestCuPyDenseDispatch:
    """ Tests if the methods and conversions have been
        succesfully registered to QuTiP's Data Layer."""

    def test_conversion_cycle(self, shape):

        from qutip.core import data

        qutip_dense = data.Dense(np.random.uniform(size=shape))

        tr1 = data.to(CuPyDense, qutip_dense)
        tr2 = data.to(data.Dense, tr1)

        np.testing.assert_array_equal(qutip_dense.to_array(), tr2.to_array())

class TestCuPyDense:
    """ Tests of the methods and constructors of the CuPyDense class. """

    def test_shape(self, shape):
        cupy_dense = CuPyDense(np.random.uniform(size=shape))
        assert (cupy_dense.shape == shape)

    def test_transpose(self, shape):
        cupy_dense = CuPyDense(np.random.uniform(size=shape)).transpose()
        np.testing.assert_array_equal(cupy_dense.shape, (shape[1], shape[0]))

    def test_adjoint(self, shape):
        data = np.random.uniform(size=shape) + 1.j*np.random.uniform(size= shape)
        cpd_adj = CuPyDense(data).adjoint()
        np.testing.assert_array_equal(cpd_adj.to_array(), data.transpose().conj())

    @pytest.mark.parametrize(["matrix", "trace"], [pytest.param([[0, 1], [1, 0]], 0),
                                                pytest.param([[2.j, 1], [1, 1]], 1+2.j)])
    def test_trace(self, matrix, trace):
        cupy_array = CuPyDense(matrix)
        assert cupy_array.trace() == trace

import time
@pytest.mark.benchmark(
    min_rounds=5,
    timer=time.time,
)
def test_true_div(shape, benchmark):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)
    
    def divide_by_2(cp_arr):
        return cp_arr /2.
    cup_arr = CuPyDense(array)
    cpdense_tr = benchmark(divide_by_2, cup_arr)
    qtpdense_tr = data.Dense(array) /2.

    assert (cpdense_tr.to_array() == qtpdense_tr.to_array()).all()


def test_itrue_div(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    cpdense_tr = CuPyDense(array).__itruediv__(2.)
    qtpdense_tr = data.Dense(array).__itruediv__(2.)

    assert (cpdense_tr.to_array() == qtpdense_tr.to_array()).all()


def test_mul(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    cpdense_tr = CuPyDense(array).__mul__(2.+1.j)
    qtpdense_tr = data.Dense(array).__mul__(2.+1.j)

    assert (cpdense_tr.to_array() == qtpdense_tr.to_array()).all()


def test_matmul(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    cpdense_tr = CuPyDense(array).__mul__(2.+1.j)
    qtpdense_tr = data.Dense(array).__mul__(2.+1.j)

    assert (cpdense_tr.to_array() == qtpdense_tr.to_array()).all()
