from __future__ import annotations

import numpy as np
import pytest

from yassm.lib.qmc.number_base import increment_inplace


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
