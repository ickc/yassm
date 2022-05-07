"""This implements the Gerber-Chopin-2015 model.

α = params[0]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    from typing import Callable

    import numpy.typing as npt

    ΑΒΣR0 = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


@jit("f8[:, ::1](f8, i8)", nopython=True, nogil=True, cache=True)
def get_F(α: float, d: int) -> npt.NDArray[np.float64]:
    res = np.empty((d, d))
    res[0, 0] = 1.0
    # first row
    for j in range(1, d):
        res[0, j] = np.power(α, j)
    for i in range(1, d):
        for j in range(d):
            exp = np.abs(i - j)
            res[i, j] = res[0, exp]
    return res


def get_ABΣR(d: int) -> tuple[ΑΒΣR0, ΑΒΣR0, ΑΒΣR0, ΑΒΣR0]:
    """Get A0, B0, Σ0, R0 callables for a given d."""

    @jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=False)
    def A0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        α = params[0]
        return get_F(α, d)

    # read-only array cannot be specified
    #  https://github.com/numba/numba/issues/4511
    @jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=False)
    def B0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.identity(d)

    return A0, B0, B0, B0


def generate_Y(T: int, d: int, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Generate Y.

    :param T: number of time step. 0 <= t <= T.
    :param d: number of dimensions.
    :param params: parameters α.
    :return: Y, shape of (T + 1, d)
    """
    α = params[0]

    Y = np.empty((T + 1, d))
    F = get_F(α, d)

    # t = 0
    X_current = np.random.randn(d)
    Y[0] = X_current + np.random.randn(d)
    for t in range(1, T + 1):
        X_current[:] = F @ X_current + np.random.randn(d)
        Y[t] = X_current + np.random.randn(d)
    return Y
