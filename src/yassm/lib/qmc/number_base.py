"""Decomposition of number in base n.

Here we use the little-endian convention.

Since we need for full sequence,
we will focus on incrementing efficiently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt


#: first 63 prime numbers.
PRIMES = [
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
    233,
    239,
    241,
    251,
    257,
    263,
    269,
    271,
    277,
    281,
    283,
    293,
    307,
]


@jit("i8(i8, i8)", nopython=True, nogil=True, cache=True)
def n_digits(N: int, base: int) -> int:
    """Return the number of digits of N in base base.

    Args:
        N (int): A positive integer to be written in base base.
        base (int): base of the number.

    Returns:
        int: The number of digits of N in base base.
    """
    return int(np.ceil(np.log(N + 1) / np.log(base)))


@jit(["(u1[::1], u1)", "(u2[::1], u2)"], nopython=True, nogil=True, cache=True)
def increment_inplace(array: npt.NDArray[np.integer[Any]], base: int) -> None:
    """Increment the array inplace.

    :param array: A number in base n in little-endian convention.
    :param base: The base of the number.
    :return: None.

    This does not check if the most significant bit is overflow.
    """
    n_digits = array.size
    # increment
    array[0] += 1
    # check overflow
    for i in range(n_digits - 1):
        if array[i] == base:
            array[i] = 0
            array[i + 1] += 1
        else:
            break
