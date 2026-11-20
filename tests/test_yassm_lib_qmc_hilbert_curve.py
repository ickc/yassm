from __future__ import annotations

import itertools

import numpy as np
import pytest

from yassm.lib.qmc.hilbert_curve import FLOAT_ORDER, hilbert_sort_inplace
from yassm.lib.qmc.hilbert_curve_full_range import hilbert_sort_inplace_func
from yassm.qmc.hilbert_curve import HilbertCurveFloat01, HilbertCurveInt


def test_hilbert_sort() -> None:
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


@pytest.mark.parametrize(
    "N, d",
    (
        (1000, 2),
        (1000, 3),
        (1000, 4),
        (1000, 5),
    ),
)
def test_hilbert_sort_float(N, d) -> None:
    array = np.random.uniform(size=(N, d))
    h = HilbertCurveFloat01(array)
    hed = h.sorted()
    h.sort()
    np.testing.assert_allclose(hed.array, h.array)


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
            # the answer returns i, which leaves A[i] untested, in the second part, when the loop over i ran into j
            return i + 1 if BitTest(A[i][ax], od) == di else i

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
    rng = np.random.default_rng(20220501)
    # make sure at most m-bit
    array = rng.integers(0, 1 << m, size=(N, n), dtype=np.uint64)
    array_list = array.tolist()
    HSort(array_list, 0, N - 1, m - 1, 0, 0, 0, False, 0)
    hilbert_sort_inplace(array, m)
    np.testing.assert_array_equal(array, array_list)


@pytest.mark.parametrize("dtype", (np.uint64, np.float64))
@pytest.mark.parametrize(
    "d, order",
    (
        (2, 1),
        (2, 3),
        (2, 6),
        (3, 2),
        (3, 4),
        (4, 3),
    ),
)
def test_hilbert_sort_subset_follows_the_curve(dtype, d, order) -> None:
    """Any subset of the grid is sorted in the order of the curve, i.e. of the sorted full grid.

    The partition used to put the last point of a group in its second part without testing it,
    so that e.g. (0, 1) came before (0, 0), although every split of a full grid, being balanced, was right.
    """
    side = 1 << order
    grid = np.array(list(itertools.product(range(side), repeat=d)), dtype=np.uint64)
    hilbert_sort_inplace(grid, order)
    # the curve visits every cell once, and moves to a neighbouring cell at each step
    assert np.unique(grid, axis=0).shape == grid.shape
    assert np.all(np.abs(np.diff(grid.astype(np.int64), axis=0)).sum(axis=1) == 1)
    rank = {tuple(row): i for i, row in enumerate(grid.tolist())}

    rng = np.random.default_rng(20220501)
    for _ in range(50):
        size = rng.integers(2, min(40, grid.shape[0]) + 1)
        subset = np.ascontiguousarray(grid[rng.choice(grid.shape[0], size=size, replace=False)])
        expected = sorted(subset.tolist(), key=lambda row: rank[tuple(row)])
        if dtype == np.float64:
            # the centres of the cells, in [0, 1)
            subset_float = (subset + 0.5) / side
            hilbert_sort_inplace(subset_float, FLOAT_ORDER)
            subset = np.floor(subset_float * side).astype(np.uint64)
        else:
            hilbert_sort_inplace(subset, order)
        np.testing.assert_array_equal(subset, expected)


@pytest.mark.parametrize("dtype", (np.uint64, np.float64))
@pytest.mark.parametrize("N", (0, 1))
def test_hilbert_sort_trivial_sizes(dtype, N) -> None:
    array = np.ones((N, 3), dtype=dtype) / 2 if dtype == np.float64 else np.ones((N, 3), dtype=dtype)
    expected = array.copy()
    hilbert_sort_inplace(array, 3 if dtype == np.uint64 else FLOAT_ORDER)
    np.testing.assert_array_equal(array, expected)


def test_hilbert_sort_order_0_is_noop() -> None:
    """Order 0 means a single cell: the order used to wrap around through int8 and shuffle the array."""
    array = np.random.default_rng(0).integers(0, 4, size=(1000, 4)).astype(np.uint64)
    expected = array.copy()
    hilbert_sort_inplace(array, 0)
    np.testing.assert_array_equal(array, expected)


def test_hilbert_curve_int_all_zeros() -> None:
    h = HilbertCurveInt(np.zeros((10, 2), dtype=np.uint64))
    assert h.order == 0
    h.sort()
    np.testing.assert_array_equal(h.array, 0)


@pytest.mark.parametrize("maximum", (1, 2, 3, 4, 2**52 - 1, 2**52, 2**60, 2**64 - 1))
def test_hilbert_curve_int_auto_order(maximum) -> None:
    """The order is the number of bits of the maximum; log2 in floating point gave one too few from 2^52."""
    array = np.array([[0, 1], [maximum, 0]], dtype=np.uint64)
    assert HilbertCurveInt(array.copy()).order == maximum.bit_length()
    with pytest.raises(ValueError, match="order"):
        HilbertCurveInt(array.copy(), order=maximum.bit_length() - 1)
    # the top bit is sorted on: (0, 1) is in the first quadrant of the curve, (maximum, 0) in the last
    h = HilbertCurveInt(array[::-1].copy())
    h.sort()
    np.testing.assert_array_equal(h.array, array)


@pytest.mark.parametrize("method", sorted(hilbert_sort_inplace_func))
def test_full_range_sort_single_point(method) -> None:
    array = np.array([[1.0, 2.0, 3.0]])
    hilbert_sort_inplace_func[method](array, FLOAT_ORDER)
    np.testing.assert_array_equal(array, [[1.0, 2.0, 3.0]])


@pytest.mark.parametrize("dimension", [2, 3])
def test_plot_axes(dimension: int) -> None:
    """The curve is drawn on a single axes: the given one, or a new one of the right projection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = HilbertCurveFloat01(np.random.default_rng(0).random((16, dimension))).sorted()

    fig, ax = curve.plot()
    assert fig.axes == [ax]
    assert ax.name == ("rectilinear" if dimension == 2 else "3d")
    plt.close(fig)

    fig = plt.figure()
    given = fig.add_subplot(projection=None if dimension == 2 else "3d")
    # the figure of the given axes is not the current one
    other = plt.figure()
    assert curve.plot(ax=given) == (fig, given)
    assert len(given.lines) == 1
    assert other.axes == []
    plt.close(fig)
    plt.close(other)


def test_plot_3d_requires_3d_axes() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="3D axes"):
        HilbertCurveFloat01(np.random.default_rng(0).random((16, 3))).plot(ax=ax)
    plt.close(fig)
