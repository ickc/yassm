from __future__ import annotations

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
    assert curve.plot(ax=given) == (fig, given)
    assert len(given.lines) == 1
    plt.close(fig)


def test_plot_3d_requires_3d_axes() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="3D axes"):
        HilbertCurveFloat01(np.random.default_rng(0).random((16, 3))).plot(ax=ax)
    plt.close(fig)
