from qutip_cupy.cudense.utils import _compare_hilbert
import pytest
import numpy as np


def ids_name(ins):
    raise ValueError(ins)

@pytest.mark.parametrize(["left", "right", "expected"], [
    pytest.param((2,), (-2,), (2,), id=f"{(2,)} x {(-2,)}"),
    pytest.param((2, 2), (-4,), (2, 2), id=f"{(2, 2)} x {(-4,)}"),
    pytest.param((2, -2), (-4,), (2, -2), id=f"{(2, -2)} x {(-4,)}"),
    pytest.param((-2, -4), (-4, -2), (-2, -2, -2), id=f"{(-2, -4)} x {(-4, -2)}"),
    pytest.param((2, -4), (4, -2), False, id=f"{(2, -4)} x {(4, -2)}"),
    pytest.param((2, -4), (-4, 2), (2, -2, 2), id=f"{(2, -4)} x {(-4, 2)}"),
    pytest.param((5,), (4,), False, id=f"{(5,)} x {(4,)}"),
    pytest.param((4, -4, 4, -4), (4, 2, -32), (4, 2, -2, 4, -4), id=f"{(4, -4, 4, -4)} x {(4, 2, -32)}"),
    pytest.param((-20, -3), (5, 4, 3), (5, 4, 3), id=f"{(-20, -3)} x {(5, 4, 3)}"),
])
def test_compare_hilbert(left, right, expected):
    if not expected:
        assert not _compare_hilbert(left, right, True)
        return
        
    out, shifts_left, shifts_right = _compare_hilbert(left, right, True)
    assert expected == out
    print(out)
    print(shifts_left, left)
    print(shifts_right, right)

    assert len(left) == len(shifts_left)
    assert sum(shifts_left.values(), start=[]) == list(range(len(out)))
    assert list(shifts_left.keys()) == list(range(len(left)))
    for size, shift in zip(left, shifts_left.values()):
        if len(shift) != 1:
            assert size < 0
        assert abs(size) == abs(np.prod([out[i] for i in shift]))

    assert len(right) == len(shifts_right)
    assert sum(shifts_right.values(), start=[]) == list(range(len(out)))
    assert list(shifts_right.keys()) == list(range(len(right)))
    for size, shift in zip(right, shifts_right.values()):
        if len(shift) != 1:
            assert size < 0
        assert abs(size) == abs(np.prod([out[i] for i in shift]))
