from __future__ import annotations

import numpy as np
from numba import jit


@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True, cache=True)
def A(params, t):
    return np.array(
        [
            [2.0, -1.0],
            [1.0, 0.0],
        ]
    )


@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True, cache=True)
def B(params, t):
    return np.array([[1.0, 0.0]])


# state cov.
@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True, cache=True)
def Σ(params, t):
    res = np.zeros((2, 2))
    res[0, 0] = params[1]
    return res


# obs. cov.
@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True, cache=True)
def R(params, t):
    return np.array([[params[0]]])
