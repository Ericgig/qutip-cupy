from qutip_cupy.cudense.utils import _compare_hilbert

@pytest.marks.parametrize(["left", "right", "expected"], [
    pytest.param((2,), (-2,), (2,)),
    pytest.param((2, 2), (-4,), (2, 2)),
    pytest.param((2, -2), (-4,), (2, -2)),
    pytest.param((-2, -4), (-4, -2), (-2, -2, -2)),
    pytest.param((2, -4), (4, -2), False),
    pytest.param((2, -4), (-4, 2), (-2, 2, -2)),
    pytest.param((5,), (4,), False),
    pytest.param((4, -4, 4, -4), (4, 2, -32), (4, 2, -2, 4, -4)),
    pytest.param((-20, -3), (5, 4, 3), (5, 4, 3)),
])
def test_compare_hilbert(left, right, expected):
    out, shifts_left, shifts_right = _compare_hilbert(left, right, True)
    assert expected == out

    assert len(left) == len(shifts_left)
    assert sum(shifts_left, start=[]) == list(range(len(out)))
    for size, shift in zip(left, shifts_left):
        if len(shift) != 1:
            assert size > 0
        assert abs(size) == abs(np.prod([out[i] for i in shift]))

    assert len(right) == len(shifts_right)
    assert sum(shifts_right, start=[]) == list(range(len(out)))
    for size, shift in zip(right, shifts_right):
        if len(shift) != 1:
            assert size > 0
        assert abs(size) == abs(np.prod([out[i] for i in shift]))
