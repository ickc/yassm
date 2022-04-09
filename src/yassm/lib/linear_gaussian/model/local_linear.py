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


@jit("(f8[::1], f8[::1], f8[::1])", nopython=True, nogil=True, cache=True)
def _sampling_parameters_1d_inplace(
    params: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> None:
    """Sampling parameters given Xs, Ys.

    Step 2 of Gibbs sampling.
    """
    size = y.size
    T = size - 1
    # x is from t = -1 to T
    params[0] = 1.0 / np.random.gamma(0.5 * size, 2.0 / np.square(y - x[1:]).sum(), size=1)[0]
    params[1] = 1.0 / np.random.gamma(0.5 * T, 2.0 / np.square(np.diff(x, n=2)).sum(), size=1)[0]


@jit("f8[::1](f8[:, ::1], f8[:, ::1])", nopython=True, nogil=True, cache=True)
def sampling_parameters(
    y_2d: npt.NDArray[np.float64],
    x_2d: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Sampling parameters given Xs, Ys.

    Step 2 of Gibbs sampling.
    """
    y = y_2d.flatten()
    size = y.size
    T = size - 1
    # x is from t = -1 to T
    x = np.empty(T + 2)
    x[0] = x_2d[0, 1]
    x[1:] = x_2d[:, 0]
    params = np.empty(2)
    _sampling_parameters_1d_inplace(params, y, x)
    return params
