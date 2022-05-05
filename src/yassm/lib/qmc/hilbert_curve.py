"""Fast Hilbert Sort Algorithm Without Using Hilbert Indices.

Sort array of uint64 integers using Hilbert sort.

Notes
-----

0 <= order < 64

Since bit is used on partition_axis/current_axis_division,
dimension < 64.

0 <= partition_axis < dimension < 64
0 <= axis_division, current_axis_division < dimension < 64
0 <= axis_start, current_axis_start, axis_counter < dimension < 64
0 <= n_cont_direction <= axis_counter < dimension < 64

Algorithm loop order from m - 1 to 1 asynchronously.

Warning
-------

Somehow when `cache=False`,
there will be segmantation fault.

TODO: use other Numba/Python versions to see if there is any difference.

Types
-----

array: u8
# index type
i_start, i_end, current_partition: i8
order: u1
axis_counter: u1
# involved in _bitset_flip
axis_start, current_axis_start: u8
axis_division, current_axis_division: 11
n_cont_direction: i1

Reference
---------

Imamura, Yasunobu, Takeshi Shinohara, Kouichi Hirata, and Tetsuji Kuboyama. “Fast Hilbert Sort Algorithm Without Using Hilbert Indices.” In Similarity Search and Applications, edited by Laurent Amsaleg, Michael E. Houle, and Erich Schubert, 9939:259–67. Lecture Notes in Computer Science. Cham: Springer International Publishing, 2016. https://doi.org/10.1007/978-3-319-46759-7_20.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    import numpy.typing as npt


@jit(
    "b1(u8, i1)",
    nopython=True,
    nogil=True,
    cache=False,
)
def _bitset_test(bits: int, position: int) -> bool:
    """Returns the value of the bit at the position pos.

    Similar to bitset::test in C++.
    """
    return bool(bits & (1 << position))


@jit(
    "u8(u8, i1)",
    nopython=True,
    nogil=True,
    cache=False,
)
def _bitset_flip(bits: int, position: int) -> int:
    """Flips the bit at the position pos.

    Similar to bitset::flip in C++.
    """
    bits ^= 1 << position
    return bits


@jit(
    "i8("
    "u8[:, :], "  # array
    "i8, "  # i_start
    "i8, "  # i_end
    "i1, "  # order
    "u8, "  # axis
    "b1, "  # direction
    ")",
    nopython=True,
    nogil=True,
    cache=False,
)
def _partition_inplace(
    array: npt.NDArray[np.uint64],
    i_start: int,
    i_end: int,
    order: int,
    axis: int,
    direction: bool,
) -> int:
    """The function partition arranges data in a similar way as the famous partition function in quick sort.

    :param array: The array to be sorted.
    :param i_start: The start index of the array.
    :param i_end: The end index of the array.
    :param order: The order of interest.
    :param axis: The axis number to be divided.
    :param direction: False for ascending, True for descending.
    """
    N, d = array.shape
    temp = np.empty(d, dtype=array.dtype)

    i = i_start
    j = i_end
    while True:
        while i < j and _bitset_test(array[i, axis], order) is direction:
            i = i + 1
        while i < j and _bitset_test(array[j, axis], order) is not direction:
            j = j - 1

        if j <= i:
            # partition finished
            return i
        # swap
        temp[:] = array[i]
        array[i] = array[j]
        array[j] = temp


@jit(
    "("
    "u8[:, ::1], "  # array
    "i8, "  # i_start
    "i8, "  # i_end
    "i1, "  # order
    "i1, "  # axis_counter
    "u8, "  # axis_start
    "i1, "  # axis_division
    "b1, "  # direction
    "i1, "  # n_cont_direction
    ")",
    nopython=True,
    nogil=True,
    cache=False,
)
def _hilbert_sort_inplace(
    array: npt.NDArray[np.uint64],
    i_start: int,
    i_end: int,
    order: int,
    axis_counter: int,
    axis_start: int,
    axis_division: int,
    direction: bool,
    n_cont_direction: int,
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
    current_partition = _partition_inplace(
        array,
        i_start,
        i_end,
        order,
        partition_axis,
        _bitset_test(axis_start, partition_axis),
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
        )


@jit(
    "(u8[:, ::1], i1)",
    nopython=True,
    nogil=True,
    cache=False,
)
def hilbert_sort_inplace(
    array: npt.NDArray[np.uint64],
    order: int,
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
    )
