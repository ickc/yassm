"""QMC point-sets or sequences as pseudo-random number generators.

All point-sets or sequences are Pseudo Unif(0, 1) exclusive.

We do not include 0 to avoid singularity,
where the point-set/sequence would be more symmetric too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    import numpy.typing as npt

from .number_base import increment_inplace, n_digits, radical_inverse


@jit("f8[::1](i8)", nopython=True, nogil=True, cache=True)
def midpoint_pointset(N: int) -> npt.NDArray[np.float64]:
    """Returns the midpoint point-set which minimize star-discrepancy in 1D."""
    res = np.empty(N)
    inv_2N = 1.0 / (2 * N)
    return np.linspace(inv_2N, 1.0 - inv_2N, N)


@jit("f8[::1](i8, i1)", nopython=True, nogil=True, cache=True)
def _van_der_corput_sequence_u1_slow(N: int, base: int) -> npt.NDArray[np.float64]:
    """Returns the van der Corput sequence."""
    res = np.empty(N)
    n = n_digits(N, base)
    array = np.zeros(n, dtype=np.uint8)
    for i in range(N):
        # we don't use 0 to avoid singularity
        # so we start from 1
        increment_inplace(array, base)

        res[i] = radical_inverse(array, base)
    return res


@jit("f8[::1](i8, i1)", nopython=True, nogil=True, cache=True)
def _van_der_corput_sequence_u2_slow(N: int, base: int) -> npt.NDArray[np.float64]:
    """Returns the van der Corput sequence."""
    res = np.empty(N)
    n = n_digits(N, base)
    array = np.zeros(n, dtype=np.uint16)
    for i in range(N):
        # we don't use 0 to avoid singularity
        # so we start from 1
        increment_inplace(array, base)

        res[i] = radical_inverse(array, base)
    return res


@jit("f8[::1](i8, i1)", nopython=True, nogil=True, cache=True)
def _van_der_corput_sequence_u1(N: int, base: int) -> npt.NDArray[np.float64]:
    """Returns the van der Corput sequence."""
    res = np.empty(N)

    n = n_digits(N, base)
    array = np.zeros(n, dtype=np.uint8)

    most_significant_bit_scale = base ** (n - 1)
    overall_scale = 1.0 / (base * most_significant_bit_scale)
    temp: int = 0
    for i in range(N):
        # we don't use 0 to avoid singularity
        # so we start from 1

        # increment
        array[0] += 1
        current_scale = most_significant_bit_scale
        temp += current_scale
        # check overflow
        for j in range(n - 1):
            if array[j] == base:
                array[j] = 0
                array[j + 1] += 1
                temp -= base * current_scale
                current_scale //= base
                temp += current_scale
            else:
                break

        res[i] = temp * overall_scale
    return res


@jit("f8[::1](i8, i1)", nopython=True, nogil=True, cache=True)
def _van_der_corput_sequence_u2(N: int, base: int) -> npt.NDArray[np.float64]:
    """Returns the van der Corput sequence."""
    res = np.empty(N)

    n = n_digits(N, base)
    array = np.zeros(n, dtype=np.uint16)

    most_significant_bit_scale = base ** (n - 1)
    overall_scale = 1.0 / (base * most_significant_bit_scale)
    temp: int = 0
    for i in range(N):
        # we don't use 0 to avoid singularity
        # so we start from 1

        # increment
        array[0] += 1
        current_scale = most_significant_bit_scale
        temp += current_scale
        # check overflow
        for j in range(n - 1):
            if array[j] == base:
                array[j] = 0
                array[j + 1] += 1
                temp -= base * current_scale
                current_scale //= base
                temp += current_scale
            else:
                break

        res[i] = temp * overall_scale
    return res


@jit("f8[::1](i8, i1)", nopython=True, nogil=True, cache=True)
def van_der_corput_sequence(N: int, base: int) -> npt.NDArray[np.float64]:
    """Returns the van der Corput sequence."""
    return _van_der_corput_sequence_u1(N, base) if base <= 255 else _van_der_corput_sequence_u2(N, base)


@jit("f8[::1](i8, i1)", nopython=True, nogil=True, cache=True)
def van_der_corput_sequence_slow(N: int, base: int) -> npt.NDArray[np.float64]:
    """Returns the van der Corput sequence.

    `van_der_corput_sequence` is a version for this function
    with `increment_inplace` and `radical_inverse` inlined
    for performance.

    This slow version is for testing purpose only.
    """
    return _van_der_corput_sequence_u1_slow(N, base) if base <= 255 else _van_der_corput_sequence_u2_slow(N, base)
