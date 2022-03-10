from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    import numpy.typing as npt


A0 = np.array(
    [
        [2.0, -1.0],
        [1.0, 0.0],
    ]
)
B0 = np.array([[1.0, 0.0]])
Σ0 = np.zeros((2, 2))
R0 = np.empty((1, 1))


# state cov.
@jit("(f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def Σ(out: npt.NDArray[np.float64], params: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> None:
    out[0, 0] = params[1]


# obs. cov.
@jit("(f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def R(out: npt.NDArray[np.float64], params: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> None:
    out[0, 0] = params[0]
