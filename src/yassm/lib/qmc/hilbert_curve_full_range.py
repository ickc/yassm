"""Fast Hilbert Sort Algorithm Without Using Hilbert Indices.

A full-range float version of Hilbert Sort.
See `.hilbert_curve` for details.

The only change here is that we no longer assumes the float is in the range of (0, 1) / [0, 1).

get_f8_to_u8 returns a function that perform the mapping from float to uint64, in 2 steps:

1.  maps the data to (0, 1) or [0, 1] in float64
2.  maps that to [0, 2^FLOAT_ORDER - 1] in uint64
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numba import jit
from scipy import linalg

from .hilbert_curve import FLOAT_ORDER, _bitset_flip, _bitset_test
from .random import correlation_variance_decomposition

if TYPE_CHECKING:
    from typing import Callable

    import numpy.typing as npt

SQRT2_INV = np.power(2.0, -0.5)


# _f8_to_u8 ####################################################################


def get_f8_to_u8(
    data: npt.NDArray[np.float64],
    method: str = "",
) -> Callable[[npt.NDArray[np.float64], int], np.uint64]:
    """Get the function to convert float64 to uint64.

    :param data: The data to be converted with unretricted range.
    :param method: The method to be used to map float to int.
        linear: linear rescaling the range.
        logistic: logistic according to gerber-chopin-2015.
        whiten_cholesky, whiten_ZCA, whiten_ZCA_cor:
            whitening then to uniform using their respective method.
    :return: The function of (x, axis) -> uint64,
        where x is the current data point of shape (d,),
        where axis is the i-th dimension of data in interested.

    This should performs 2 steps,

    1.  maps the data to (0, 1) or [0, 1] in float64
    2.  maps that to [0, 2^FLOAT_ORDER - 1] in uint64
    """

    if method == "linear":

        low = data.min(axis=0)
        high = data.max(axis=0)
        scale = np.reciprocal(high - low)

        @jit("u8(f8[::1], i8)", nopython=True, nogil=True, cache=False)
        def f8_to_u8(x: npt.NDArray[np.float64], i: int) -> int:
            if x[i] == high[i]:
                return 1 << FLOAT_ORDER - 1
            # this transformation goes to [0, 1],
            # but 1 is special-cased above.
            # i.e. to [0, 1)
            x_i = scale[i] * (x[i] - low[i])
            # maps to [0, 2^FLOAT_ORDER - 1] evenly
            return np.floor(x_i * (1 << FLOAT_ORDER))

    elif method == "logistic":

        mean = data.mean(axis=0)
        std = data.std(axis=0, ddof=1)

        low = mean - 2.0 * std
        high = mean + 2.0 * std
        # 0.5 comes from tanh(y / 2)
        scale = 0.5 / (high - low)

        @jit("u8(f8[::1], i8)", nopython=True, nogil=True, cache=False)
        def f8_to_u8(x: npt.NDArray[np.float64], i: int) -> int:
            # to (0, 1)
            x_i = 0.5 + 0.5 * np.tanh(scale[i] * (x[i] - low[i]))
            # maps to [0, 2^FLOAT_ORDER - 1] evenly
            return np.floor(x_i * (1 << FLOAT_ORDER))

    elif method.startswith("whiten_"):

        mean = data.mean(axis=0)
        cov = np.cov(data, rowvar=False)

        if method == "whiten_cholesky":
            W_T = linalg.cholesky(linalg.pinv(cov), lower=True)
        elif method == "whiten_ZCA":
            W_T = linalg.pinv(linalg.sqrtm(cov))
        elif method == "whiten_ZCA_cor":
            σ, P = correlation_variance_decomposition(cov)
            P_sqrt_inv = linalg.pinv(linalg.sqrtm(P))
            W_T = P_sqrt_inv / σ.reshape(-1, 1)
        else:
            raise ValueError(f"Unknown method: {method}")

        @jit("u8(f8[::1], i8)", nopython=True, nogil=True, cache=False)
        def f8_to_u8(x: npt.NDArray[np.float64], i: int) -> int:
            # whitening
            G = (x - mean) @ W_T
            # to (0, 1)
            x_i = 0.5 + 0.5 * math.erf(G[i] * SQRT2_INV)
            # maps to [0, 2^FLOAT_ORDER - 1] evenly
            return np.floor(x_i * (1 << FLOAT_ORDER))

    else:
        raise ValueError(f"Unknown method: {method}")

    return f8_to_u8


# for arbitrary range of float #################################################


@jit(nopython=True, nogil=True, cache=False)
def _partition_inplace_float(
    array: npt.NDArray[np.float64],
    i_start: int,
    i_end: int,
    order: int,
    axis: int,
    direction: bool,
    f8_to_u8: Callable[[npt.NDArray[np.float64], int], np.uint64],
) -> int:
    """The function partition arranges data in a similar way as the famous partition function in quick sort.

    :param array: The array to be sorted.
    :param i_start: The start index of the array.
    :param i_end: The end index of the array.
    :param order: The order of interest.
    :param axis: The axis number to be divided.
    :param direction: False for ascending, True for descending.
    :param f8_to_u8: The function to convert float to uint64.
    """
    N, d = array.shape
    temp = np.empty(d, dtype=array.dtype)

    i = i_start
    j = i_end
    while True:
        while i < j and _bitset_test(f8_to_u8(array[i], axis), order) is direction:
            i = i + 1
        while i < j and _bitset_test(f8_to_u8(array[j], axis), order) is not direction:
            j = j - 1

        if j <= i:
            # partition finished
            return i
        # swap
        temp[:] = array[i]
        array[i] = array[j]
        array[j] = temp


@jit(nopython=True, nogil=True, cache=False)
def _hilbert_sort_inplace(
    array: npt.NDArray[np.float64],
    i_start: int,
    i_end: int,
    order: int,
    axis_counter: int,
    axis_start: int,
    axis_division: int,
    direction: bool,
    n_cont_direction: int,
    f8_to_u8: Callable[[npt.NDArray[np.float64], int], np.uint64],
) -> None:
    """Perform Hilbert sort on the array inplace.

    :param array: The array to be sorted.
    :param i_start: The start index of the array.
    :param i_end: The end index of the array.
    :param order: The current order of the Hilbert curve.
    :param axis_counter: Axis counter in current order.
    :param axis_start: Start positions of axes.
    :param axis_division: Axis number of 1st division on current order.
    :param direction: Direction of previous division. False for ascending, True for descending.
    :param n_cont_direction: Number of continuous occurrences of direction at same order.
    :param f8_to_u8: The function to convert float to uint64.
    """
    dimension = array.shape[1]

    partition_axis: int
    current_partition: int
    current_axis_division: int
    current_axis_start: int

    if i_end <= i_start:
        # nothing to do for empty or singleton array
        return
    partition_axis = (axis_counter + axis_division) % dimension
    current_partition = _partition_inplace_float(
        array,
        i_start,
        i_end,
        order,
        partition_axis,
        _bitset_test(axis_start, partition_axis),
        f8_to_u8,
    )

    if axis_counter == dimension - 1:
        # simulation done, goto next order
        if order == 0:
            # no more order to simulate
            return
        order -= 1

        current_axis_division = (axis_division + 2 * dimension - (2 if direction else n_cont_direction + 2)) % dimension
        current_axis_start = _bitset_flip(axis_start, current_axis_division)
        current_axis_start = _bitset_flip(current_axis_start, partition_axis)
        _hilbert_sort_inplace(
            array,
            i_start,
            current_partition - 1,
            order,
            0,
            current_axis_start,
            current_axis_division,
            False,
            0,
            f8_to_u8,
        )

        current_axis_start = axis_start
        current_axis_division = (axis_division + 2 * dimension - (n_cont_direction + 2 if direction else 2)) % dimension
        _hilbert_sort_inplace(
            array,
            current_partition,
            i_end,
            order,
            0,
            current_axis_start,
            current_axis_division,
            False,
            0,
            f8_to_u8,
        )
    else:
        axis_counter += 1

        current_axis_start = axis_start
        _hilbert_sort_inplace(
            array,
            i_start,
            current_partition - 1,
            order,
            axis_counter,
            current_axis_start,
            axis_division,
            False,
            1 if direction else n_cont_direction + 1,
            f8_to_u8,
        )

        current_axis_start = _bitset_flip(axis_start, partition_axis)
        # new partition axis_start after updating axis_counter
        partition_axis = (axis_counter + axis_division) % dimension
        current_axis_start = _bitset_flip(current_axis_start, partition_axis)
        _hilbert_sort_inplace(
            array,
            current_partition,
            i_end,
            order,
            axis_counter,
            current_axis_start,
            axis_division,
            True,
            n_cont_direction + 1 if direction else 1,
            f8_to_u8,
        )


@jit(nopython=True, nogil=True, cache=False)
def hilbert_sort_inplace(
    array: npt.NDArray[np.float64],
    order: int,
    f8_to_u8: Callable[[npt.NDArray[np.float64], int], np.uint64],
) -> None:
    """Perform Hilbert sort on the array inplace.

    :param array: The array to be sorted, dtype of uint64, shape of (N, d),
        where N is the number of data, and d is the dimension of data.
    :param order: The order of the Hilbert curve, 1 <= order <= 64.
    """
    N = array.shape[0]
    _hilbert_sort_inplace(
        array,
        0,
        N - 1,
        order - 1,
        0,
        0,
        0,
        False,
        0,
        f8_to_u8,
    )
