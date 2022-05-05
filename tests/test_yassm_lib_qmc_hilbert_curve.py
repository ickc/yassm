from __future__ import annotations

import numpy as np
import pytest

from yassm.lib.qmc.hilbert_curve import hilbert_sort_inplace


def test_hilbert_sort() -> None:
    N = 9  # 9 points
    m = 3  # order of Hilbert curve
    array = np.array(
        [
            [2, 2],
            [2, 4],
            [3, 4],
            [2, 5],
            [3, 5],
            [1, 6],
            [3, 6],
            [5, 6],
            [3, 7],
        ],
        dtype=np.uint64,
    )
    hilbert_sort_inplace(array, m)
    np.testing.assert_array_equal(
        array,
        [
            [2, 2],
            [1, 6],
            [3, 7],
            [3, 6],
            [3, 5],
            [2, 5],
            [2, 4],
            [3, 4],
            [5, 6],
        ],
    )


# compare to https://stackoverflow.com/a/66994116/5769446


def BitTest(x, od):
    result = x & (1 << od)
    return int(bool(result))


def BitFlip(b, pos):
    b ^= 1 << pos
    return b


def partition(A, st, en, od, ax, di):
    i = st
    j = en
    while True:
        while i < j and BitTest(A[i][ax], od) == di:
            i = i + 1
        while i < j and BitTest(A[j][ax], od) != di:
            j = j - 1

        if j <= i:
            return i

        A[i], A[j] = A[j], A[i]


def HSort(A, st, en, od, c, e, d, di, cnt):
    n = len(A[0])

    if en <= st:
        return
    p = partition(A, st, en, od, (d + c) % n, BitTest(e, (d + c) % n))

    if c == n - 1:
        if od == 0:
            return

        d2 = (d + n + n - (2 if di else cnt + 2)) % n
        e = BitFlip(e, d2)
        e = BitFlip(e, (d + c) % n)
        HSort(A, st, p - 1, od - 1, 0, e, d2, False, 0)

        e = BitFlip(e, (d + c) % n)
        e = BitFlip(e, d2)
        d2 = (d + n + n - (cnt + 2 if di else 2)) % n
        HSort(A, p, en, od - 1, 0, e, d2, False, 0)
    else:
        HSort(A, st, p - 1, od, c + 1, e, d, False, (1 if di else cnt + 1))
        e = BitFlip(e, (d + c) % n)
        e = BitFlip(e, (d + c + 1) % n)
        HSort(A, p, en, od, c + 1, e, d, True, (cnt + 1 if di else 1))
        e = BitFlip(e, (d + c + 1) % n)
        e = BitFlip(e, (d + c) % n)


# end https://stackoverflow.com/a/66994116/5769446


@pytest.mark.parametrize(
    "N, m, n",
    (
        (9, 3, 2),
        (9, 3, 3),
        (100, 63, 10),
    ),
)
def test_compare_to_stackoverflow(N, m, n):
    rng = np.random.default_rng()
    # make sure at most m-bit
    array = rng.integers(0, 1 << m, size=(N, n), dtype=np.uint64)
    array_list = array.tolist()
    HSort(array_list, 0, N - 1, m - 1, 0, 0, 0, False, 0)
    hilbert_sort_inplace(array, m)
    np.testing.assert_array_equal(array, array_list)
