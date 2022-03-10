from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    import numpy.typing as npt


@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True, cache=True)
def A(params: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array(
        [
            [2.0, -1.0],
            [1.0, 0.0],
        ]
    )


@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True, cache=True)
def B(params: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([[1.0, 0.0]])


# state cov.
@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True, cache=True)
def Σ(params: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    res = np.zeros((2, 2))
    res[0, 0] = params[1]
    return res


# obs. cov.
@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True, cache=True)
def R(params: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([[params[0]]])
