from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit, vectorize

if TYPE_CHECKING:
    import numpy.typing as npt


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


@vectorize("f8(f8, f8, f8)")
def normal_1d(
    x: npt.NDArray[np.float64],
    μ: npt.NDArray[np.float64],
    Σ: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    res = x - μ
    return np.sqrt(2.0 * np.pi * Σ) * np.exp(-0.5 * (np.square(res) / Σ))
