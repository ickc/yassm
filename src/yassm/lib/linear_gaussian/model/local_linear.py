from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    import numpy.typing as npt


@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def A0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array(
        [
            [2.0, -1.0],
            [1.0, 0.0],
        ]
    )


@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def B0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([[1.0, 0.0]])


# state cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def Σ0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out = np.zeros((2, 2))
    out[0, 0] = params[1]
    return out


# obs. cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def R0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out = np.empty((1, 1))
    out[0, 0] = params[0]
    return out
