from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
from numba import jit, vectorize

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


@overload
def normal_1d(  # type: ignore[misc]
    x: float,
    μ: float,
    Σ: float,
) -> float:
    ...


@overload
def normal_1d(
    x: float | npt.NDArray[np.float64],
    μ: float | npt.NDArray[np.float64],
    Σ: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    ...


@vectorize("f8(f8, f8, f8)")
def normal_1d(
    x: float | npt.NDArray[np.float64],
    μ: float | npt.NDArray[np.float64],
    Σ: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.exp(-0.5 * (np.square(x - μ) / Σ)) / np.sqrt(2.0 * np.pi * Σ)
