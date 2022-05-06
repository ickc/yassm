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

#: first 63 prime numbers.
PRIMES = np.array(
    [
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
    ],
    dtype=np.uint16,
)


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

    n: int = n_digits(N, base)
    array = np.zeros(n, dtype=np.uint8)

    most_significant_bit_scale: int = base ** (n - 1)
    overall_scale: int = base * most_significant_bit_scale
    temp: int = 0
    for i in range(N):
        # we don't use 0 to avoid singularity
        # so we start from 1

        # increment
        array[0] += 1
        current_scale: int = most_significant_bit_scale
        current_scale_times_base: int = overall_scale
        temp += current_scale
        # check overflow
        for j in range(n - 1):
            if array[j] == base:
                array[j] = 0
                array[j + 1] += 1
                temp -= current_scale_times_base
                # decrease by a factor of base
                current_scale_times_base = current_scale
                current_scale //= base
                temp += current_scale
            else:
                break

        # maintain accuracy by doing one single lossy division at the end
        res[i] = temp / overall_scale
    return res


@jit("f8[::1](i8, i1)", nopython=True, nogil=True, cache=True)
def _van_der_corput_sequence_u2(N: int, base: int) -> npt.NDArray[np.float64]:
    """Returns the van der Corput sequence."""
    res = np.empty(N)

    n: int = n_digits(N, base)
    array = np.zeros(n, dtype=np.uint16)

    most_significant_bit_scale: int = base ** (n - 1)
    overall_scale: int = base * most_significant_bit_scale
    temp: int = 0
    for i in range(N):
        # we don't use 0 to avoid singularity
        # so we start from 1

        # increment
        array[0] += 1
        current_scale: int = most_significant_bit_scale
        current_scale_times_base: int = overall_scale
        temp += current_scale
        # check overflow
        for j in range(n - 1):
            if array[j] == base:
                array[j] = 0
                array[j + 1] += 1
                temp -= current_scale_times_base
                # decrease by a factor of base
                current_scale_times_base = current_scale
                current_scale //= base
                temp += current_scale
            else:
                break

        # maintain accuracy by doing one single lossy division at the end
        res[i] = temp / overall_scale
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


@jit("f8[:, ::1](i8, i1, b1)", nopython=True, nogil=True, cache=True)
def halton_sequence(N: int, d: int, transpose: bool) -> npt.NDArray[np.float64]:
    """Returns the Halton sequence.

    :param N: number of samples
    :param d: number of dimensions
    :param transpose: whether to transpose the result
    :return: the Halton sequence
    """
    if not (2 <= d < 64):
        raise ValueError("d must be between 2 and 63")
    res = np.empty((d, N))
    for i in range(d):
        prime = PRIMES[i]
        res[i] = van_der_corput_sequence(N, prime)
    return np.ascontiguousarray(res.T) if transpose else res


# Matrix #######################################################################


@jit("f8[:, ::1](f8, f8, i8)", nopython=True, nogil=True, cache=True)
def generate_positive_semidefinite_matrix(low: float, high: float, size: int) -> npt.NDArray[np.float64]:
    """Generates a positive semidefinite matrix.

    For testing use.
    """
    L = np.random.uniform(low, high, size=size * size).reshape(size, size)
    return L @ L.T


@jit("Tuple((f8[::1], f8[:, ::1]))(f8[:, ::1])", nopython=True, nogil=True, cache=True)
def correlation_variance_decomposition(
    Σ: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Returns the correlation and variance decomposition of a matrix.

    .. math::
        V^\frac{1}{2} = \text{diag}\{σ\}
        Σ = V^\frac{1}{2} P V^\frac{1}{2}
    """
    n = Σ.shape[0]

    temp = np.empty((n + 1, n))
    σ = temp[0]
    P = temp[1:]

    for i in range(n):
        σ[i] = np.sqrt(Σ[i, i])
    for i in range(n):
        for j in range(n):
            P[i, j] = Σ[i, j] / (σ[i] * σ[j])
    return σ, P
