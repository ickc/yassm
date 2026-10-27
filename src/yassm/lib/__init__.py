from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    from typing import TypeVar

    import numpy.typing as npt

    FloatMaybeArray = TypeVar("FloatMaybeArray", float, npt.NDArray[np.float64])


@jit(
    "f8[:, ::1](f8[:, :, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def sqrt_diag(array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    m, n, k = array.shape
    d = min(n, k)
    res = np.empty((m, d))
    for i in range(m):
        for j in range(d):
            res[i, j] = np.sqrt(array[i, j, j])
    return res


@jit("(f8[:, ::1], f8[:, ::1], i8, f8[::1])", nopython=True, nogil=True, cache=True)
def random_choice_weighted_inplace(
    choice: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    size: int,
    p: npt.NDArray[np.float64],
) -> None:
    """weighted random choice with replacement, in-place.

    Inplace version of random_choice_weighted.
    """
    b = np.cumsum(p)
    # if p is normalized, b[-1] == 1.0
    u = np.random.uniform(0.0, b[-1], size)
    # idxs = np.digitize(u, b)
    # see footnote of numpy.digitize
    # that when b is monotonic increasing, the 2 algorithm are the same
    idxs = np.searchsorted(b, u, side="right")
    n_bin = b.size
    for i in range(size):
        idx = idxs[i]
        # numerical issue might result in some u >= b[-1]
        # this is the only exception in digitize/searchsorted
        # that the index can fall outside
        choice[i] = x[n_bin - 1 if idx == n_bin else idx]


@jit("f8[:, ::1](f8[:, ::1], i8, f8[::1])", nopython=True, nogil=True, cache=True)
def random_choice_weighted(
    x: npt.NDArray[np.float64],
    size: int,
    p: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """weighted random choice with replacement.

    :param: x: 2d-array to be chosen from, shape of (n_bin, d).
    :param: size: number of samples.
    :param: p: probability of each element in x, assumes to be positive, shape of (n_bin,) without checking.

    Similar to `np.random.default_rng().choice(x, size=size, p=p)`,
    but p can be non-normalized.
    """
    d = x.shape[1]

    # memalloc
    choice = np.empty((size, d))
    random_choice_weighted_inplace(choice, x, size, p)
    return choice
