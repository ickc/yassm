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
from numba import jit, objmode
from scipy import linalg

from .hilbert_curve import FLOAT_ORDER, _bitset_flip, _bitset_test
from .random import correlation_variance_decomposition

if TYPE_CHECKING:
    from typing import Callable

    import numpy.typing as npt

    Args = tuple[npt.NDArray[np.float64], ...]
    Gen = Callable[[npt.NDArray[np.float64]], Args]
    F8ToU8 = Callable[[np.float64, int, Args], int]
    HilbertSortInplace = Callable[[npt.NDArray[np.float64], int], None]

SQRT2_INV = np.power(2.0, -0.5)


# _f8_to_u8 ####################################################################


@jit("Tuple((f8[::1], f8[::1], f8[::1]))(f8[:, ::1])", nopython=True, nogil=True, cache=True)
def gen_f8_to_u8_args_linear(data: npt.NDArray[np.float64]) -> Args:
    N, d = data.shape

    res = np.empty((3, d))
    low = res[0]
    high = res[1]
    scale = res[2]

    # initialize
    low[:] = data[0]
    high[:] = low
    for i in range(1, N):
        row = data[i]
        for j in range(d):
            datum = row[j]
            low[j] = min(datum, low[j])
            high[j] = max(datum, high[j])
    scale[:] = np.reciprocal(high - low)
    return low, high, scale


@jit("u8(f8[::1], i8, Tuple((f8[::1], f8[::1], f8[::1])))", nopython=True, nogil=True, cache=True)
def f8_to_u8_linear(x: npt.NDArray[np.float64], i: int, args: Args) -> int:
    low, high, scale = args

    if x[i] == high[i]:
        return 1 << FLOAT_ORDER - 1
    # this transformation goes to [0, 1],
    # but 1 is special-cased above.
    # i.e. to [0, 1)
    x_i = scale[i] * (x[i] - low[i])
    # maps to [0, 2^FLOAT_ORDER - 1] evenly
    return np.floor(x_i * (1 << FLOAT_ORDER))


@jit("Tuple((f8[::1], f8[::1]))(f8[:, ::1])", nopython=True, nogil=True, cache=True)
def gen_f8_to_u8_args_logistic(data: npt.NDArray[np.float64]) -> Args:
    N, d = data.shape

    mean = data.sum(axis=0) / N
    # numba doesn't support ddof
    ddof_factor_times2 = 2.0 * np.sqrt(N / (N - 1))
    std_times2 = np.empty(d)
    for j in range(d):
        std_times2[j] = ddof_factor_times2 * data[:, j].std()

    low = mean - std_times2
    high = mean + std_times2
    # 0.5 comes from tanh(y / 2)
    scale = 0.5 / (high - low)
    return low, scale


@jit("u8(f8[::1], i8, Tuple((f8[::1], f8[::1])))", nopython=True, nogil=True, cache=True)
def f8_to_u8_logistic(x: npt.NDArray[np.float64], i: int, args: Args) -> int:
    low, scale = args
    # to (0, 1)
    x_i = 0.5 + 0.5 * np.tanh(scale[i] * (x[i] - low[i]))
    # maps to [0, 2^FLOAT_ORDER - 1] evenly
    return np.floor(x_i * (1 << FLOAT_ORDER))


@jit("Tuple((f8[::1], f8[:, ::1]))(f8[:, ::1])", nopython=True, nogil=True, cache=True)
def gen_f8_to_u8_args_whiten_cholesky(data: npt.NDArray[np.float64]) -> Args:
    N, d = data.shape

    mean = data.sum(axis=0) / N
    cov = np.cov(data, rowvar=False)

    # avoid LinAlgError: not positive definite
    cov_inv = np.linalg.pinv(cov)
    # force symmetric
    cov_inv = 0.5 * (cov_inv + cov_inv.T)
    # minimum eigenvalue
    eigen_min = np.linalg.eigvalsh(cov_inv)[0]
    # force eigen_min >= 0
    if eigen_min < 0.0:
        cov_inv += eigen_min * np.eye(d)

    # W_T = L = U.T
    W_T = np.linalg.cholesky(cov_inv)
    return mean, W_T


@jit("Tuple((f8[::1], f8[:, ::1]))(f8[:, ::1])", nopython=True, nogil=True, cache=True)
def gen_f8_to_u8_args_whiten_ZCA(data: npt.NDArray[np.float64]) -> Args:
    N, d = data.shape

    mean = data.sum(axis=0) / N
    cov = np.cov(data, rowvar=False)
    with objmode(W_T_inv="f8[:, ::1]"):
        W_T_inv = linalg.sqrtm(cov)
    return mean, np.linalg.pinv(W_T_inv).T


@jit("Tuple((f8[::1], f8[:, ::1]))(f8[:, ::1])", nopython=True, nogil=True, cache=True)
def gen_f8_to_u8_args_whiten_ZCA_cor(data: npt.NDArray[np.float64]) -> Args:
    N, d = data.shape

    mean = data.sum(axis=0) / N
    cov = np.cov(data, rowvar=False)
    σ, P = correlation_variance_decomposition(cov)
    with objmode(P_sqrt="f8[:, ::1]"):
        P_sqrt = linalg.sqrtm(P)
    P_sqrt_inv = np.linalg.pinv(P_sqrt).T
    W_T = P_sqrt_inv / σ.reshape(-1, 1)
    return mean, W_T


@jit("u8(f8[::1], i8, Tuple((f8[::1], f8[:, ::1])))", nopython=True, nogil=True, cache=True)
def f8_to_u8_whiten(x: npt.NDArray[np.float64], i: int, args: Args) -> int:
    mean, W_T = args
    # whitening
    G = (x - mean) @ W_T
    # to (0, 1)
    x_i = 0.5 + 0.5 * math.erf(G[i] * SQRT2_INV)
    # maps to [0, 2^FLOAT_ORDER - 1] evenly
    return np.floor(x_i * (1 << FLOAT_ORDER))


# for arbitrary range of float #################################################


@jit(nopython=True, nogil=True, cache=False)
def _partition_inplace_float(
    array: npt.NDArray[np.float64],
    i_start: int,
    i_end: int,
    order: int,
    axis: int,
    direction: bool,
    f8_to_u8: F8ToU8,
    args: Args,
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
        while i < j and _bitset_test(f8_to_u8(array[i], axis, args), order) is direction:
            i = i + 1
        while i < j and _bitset_test(f8_to_u8(array[j], axis, args), order) is not direction:
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
    f8_to_u8: F8ToU8,
    args: Args,
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
        args,
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
            args,
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
            args,
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
            args,
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
            args,
        )


@jit("(f8[:, ::1], i1)", nopython=True, nogil=True, cache=False)
def hilbert_sort_inplace_linear(
    array: npt.NDArray[np.float64],
    order: int,
) -> None:
    """Perform Hilbert sort on the array inplace.

    :param array: The array to be sorted, dtype of uint64, shape of (N, d),
        where N is the number of data, and d is the dimension of data.
    :param order: The order of the Hilbert curve, 1 <= order <= 64.
    """
    N = array.shape[0]
    args = gen_f8_to_u8_args_linear(array)

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
        f8_to_u8_linear,
        args,
    )


@jit("(f8[:, ::1], i1)", nopython=True, nogil=True, cache=False)
def hilbert_sort_inplace_logistic(
    array: npt.NDArray[np.float64],
    order: int,
) -> None:
    """Perform Hilbert sort on the array inplace.

    :param array: The array to be sorted, dtype of uint64, shape of (N, d),
        where N is the number of data, and d is the dimension of data.
    :param order: The order of the Hilbert curve, 1 <= order <= 64.
    """
    N = array.shape[0]
    args = gen_f8_to_u8_args_logistic(array)

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
        f8_to_u8_logistic,
        args,
    )


@jit("(f8[:, ::1], i1)", nopython=True, nogil=True, cache=False)
def hilbert_sort_inplace_whiten_cholesky(
    array: npt.NDArray[np.float64],
    order: int,
) -> None:
    """Perform Hilbert sort on the array inplace.

    :param array: The array to be sorted, dtype of uint64, shape of (N, d),
        where N is the number of data, and d is the dimension of data.
    :param order: The order of the Hilbert curve, 1 <= order <= 64.
    """
    N = array.shape[0]
    args = gen_f8_to_u8_args_whiten_cholesky(array)

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
        f8_to_u8_whiten,
        args,
    )


@jit("(f8[:, ::1], i1)", nopython=True, nogil=True, cache=False)
def hilbert_sort_inplace_whiten_ZCA(
    array: npt.NDArray[np.float64],
    order: int,
) -> None:
    """Perform Hilbert sort on the array inplace.

    :param array: The array to be sorted, dtype of uint64, shape of (N, d),
        where N is the number of data, and d is the dimension of data.
    :param order: The order of the Hilbert curve, 1 <= order <= 64.
    """
    N = array.shape[0]
    args = gen_f8_to_u8_args_whiten_ZCA(array)

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
        f8_to_u8_whiten,
        args,
    )


@jit("(f8[:, ::1], i1)", nopython=True, nogil=True, cache=False)
def hilbert_sort_inplace_whiten_ZCA_cor(
    array: npt.NDArray[np.float64],
    order: int,
) -> None:
    """Perform Hilbert sort on the array inplace.

    :param array: The array to be sorted, dtype of uint64, shape of (N, d),
        where N is the number of data, and d is the dimension of data.
    :param order: The order of the Hilbert curve, 1 <= order <= 64.
    """
    N = array.shape[0]
    args = gen_f8_to_u8_args_whiten_ZCA_cor(array)

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
        f8_to_u8_whiten,
        args,
    )


gen_f8_to_u8_args_func: dict[str, Gen] = {
    "linear": gen_f8_to_u8_args_linear,
    "logistic": gen_f8_to_u8_args_logistic,
    "whiten_cholesky": gen_f8_to_u8_args_whiten_cholesky,
    "whiten_ZCA": gen_f8_to_u8_args_whiten_ZCA,
    "whiten_ZCA_cor": gen_f8_to_u8_args_whiten_ZCA_cor,
}

f8_to_u8_func: dict[str, F8ToU8] = {
    "linear": f8_to_u8_linear,
    "logistic": f8_to_u8_logistic,
    "whiten_cholesky": f8_to_u8_whiten,
    "whiten_ZCA": f8_to_u8_whiten,
    "whiten_ZCA_cor": f8_to_u8_whiten,
}

hilber_sort_inplace_func: dict[str, HilbertSortInplace] = {
    "linear": hilbert_sort_inplace_linear,
    "logistic": hilbert_sort_inplace_logistic,
    "whiten_cholesky": hilbert_sort_inplace_whiten_cholesky,
    "whiten_ZCA": hilbert_sort_inplace_whiten_ZCA,
    "whiten_ZCA_cor": hilbert_sort_inplace_whiten_ZCA_cor,
}
