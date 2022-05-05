from __future__ import annotations

import numpy as np
import pytest

from yassm.lib.qmc.number_base import (
    base_repr_u1,
    base_repr_u2,
    increment_inplace,
    int_repr,
    radical_inverse,
    radical_inverse_numpy,
)


@pytest.mark.parametrize(
    "n_digits, base",
    (
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 5),
        (3, 6),
        (3, 7),
        (3, 8),
        (3, 9),
        (3, 10),
    ),
)
def test_increment(
    n_digits: int,
    base: int,
):
    """Test increment."""
    array = np.zeros(n_digits, dtype=np.uint8)
    for i in range(base**n_digits):
        # print(f"checking number {i}")
        # check against numpy
        ref = np.base_repr(i, base=base)
        # print(array, ref)
        for position, digit in enumerate(reversed(ref)):
            # print(f"checking position {position}")
            assert int(digit) == array[position]
        last_position = position
        for position in range(last_position + 1, n_digits):
            # print(f"checking remaining digits to be zero at position {position}")
            assert array[position] == 0
        increment_inplace(array, base)


N_BASE_COMBO = (
    (869719, 2),
    (342822, 3),
    (695944, 4),
    (697910, 5),
    (221370, 6),
    (768903, 7),
    (83217, 8),
    (922337203685477580, 9),
)


@pytest.mark.parametrize("N, base", N_BASE_COMBO)
def test_base_repr(N, base):
    array = base_repr_u1(N, base)
    np.testing.assert_array_equal(base_repr_u2(N, base), array)
    assert int("".join(str(i) for i in reversed(array)), base) == N
    assert int_repr(array, base) == N


@pytest.mark.parametrize("N, base", N_BASE_COMBO)
def test_radical_inverse(N, base):
    array = base_repr_u1(N, base)
    array2 = base_repr_u2(N, base)
    ref = radical_inverse_numpy(N, base)
    res = radical_inverse(array, base)
    res2 = radical_inverse(array2, base)
    assert np.allclose(res, ref)
    assert np.allclose(res2, ref)
