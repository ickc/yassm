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


@jit("i8(i8, i8)", nopython=True, nogil=True, cache=True)
def n_digits(N: int, base: int) -> int:
    """Return the number of digits of N in base base.

    Parameters
    ----------
    N
        A positive integer to be written in base `base`.
    base
        Base of the number.

    Returns
    -------
    int
        The number of digits of N in base `base`.
    """
    return int(np.ceil(np.log(N + 1) / np.log(base)))


@jit("u1[::1](i8, i8)", nopython=True, nogil=True, cache=True)
def base_repr_u1(N: int, base: int) -> npt.NDArray[np.uint8]:
    """Return the base representation of a number in a base.

    Parameters
    ----------
    N
        The number to be represented.
    base
        The base of the number.

    Returns
    -------
    npt.NDArray[np.uint8]
        The base representation of the number in little endian convention.
    """
    n = n_digits(N, base)
    base_repr = np.empty(n, dtype=np.uint8)
    for i in range(n):
        base_repr[i] = N % base
        N //= base
    return base_repr


@jit("u2[::1](i8, i8)", nopython=True, nogil=True, cache=True)
def base_repr_u2(N: int, base: int) -> npt.NDArray[np.uint16]:
    """Return the base representation of a number in a base.

    Parameters
    ----------
    N
        The number to be represented.
    base
        The base of the number.

    Returns
    -------
    npt.NDArray[np.uint16]
        The base representation of the number in little endian convention.
    """
    n = n_digits(N, base)
    base_repr = np.empty(n, dtype=np.uint16)
    for i in range(n):
        base_repr[i] = N % base
        N //= base
    return base_repr


@jit(["i8(u1[::1], i8)", "i8(u2[::1], i8)"], nopython=True, nogil=True, cache=True)
def int_repr(array: npt.NDArray[np.integer[Any]], base: int) -> int:
    """Return the integer representation of an array in a base.

    Parameters
    ----------
    array
        The array in base representation with little endian convention.
    base
        The base of the number.

    Returns
    -------
    int
        The integer representation of the array.
    """
    n = array.size
    res: int = 0
    scale: int = 1
    for k in range(n):
        res += array[k] * scale
        scale *= base
    return res


@jit(["(u1[::1], u1)", "(u2[::1], u2)"], nopython=True, nogil=True, cache=True)
def increment_inplace(array: npt.NDArray[np.integer[Any]], base: int) -> None:
    """Increment the array inplace.

    Parameters
    ----------
    array
        A number in base n in little-endian convention.
    base
        The base of the number.

    Returns
    -------
    None
        None.

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


@jit(["f8(u1[::1], u1)", "f8(u2[::1], u2)"], nopython=True, nogil=True, cache=True)
def radical_inverse(array: npt.NDArray[np.integer[Any]], base: int) -> float:
    """Return the radical inverse of the array in a base.

    This will overflow when the array is too large,
    basically if radical_inverse * base**len(array) >= 2**63,
    i.e. if the reverse digits of array are too large to be represented in int64,
    it will overflow.

    Practically we won't be having such large numbers.
    """
    n_digits = array.size

    # use integer for exact arithmetic
    # and in the end divide by common denominator
    # 0 <= k <= n_digits - 1
    # -1 >= -1 - k >= -n_digits
    # n_digits - 1 >= n_digits - 1 - k >= 0
    res: int = 0
    scale: int = 1
    for k in range(n_digits - 1, -1, -1):
        a = array[k]
        res += a * scale
        scale *= base
    # maintain accuracy by doing one single lossy division at the end
    return res / scale


def radical_inverse_numpy(N: int, base: int) -> None:
    """Return the radical inverse of a number in a base.

    This uses numpy which support base up to 36 only.

    This is mainly used for testing.
    """
    base_repr = np.binary_repr(N) if base == 2 else np.base_repr(N, base)
    return int(base_repr[::-1], base) / base ** len(base_repr)
