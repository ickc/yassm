"""Moving average of order 1.

σ_sq, θ = params
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    import numpy.typing as npt


@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def A0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out = np.zeros((2, 2))
    out[0, 1] = 1.0
    return out


@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def B0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out = np.zeros((1, 2))
    out[0, 0] = 1.0
    return out


# state cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def Σ0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    σ_sq, θ = params
    σ_sq_θ = σ_sq * θ
    return np.array(
        [
            [σ_sq, σ_sq_θ],
            [σ_sq_θ, σ_sq_θ * θ],
        ]
    )


# obs. cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def R0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.zeros((1, 1))


# state cov.
@jit("Tuple((f8[::1], f8[:, ::1]))(f8[::1])", nopython=True, nogil=True, cache=True)
def X0(params: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    σ_sq, θ = params
    σ_sq_θ = σ_sq * θ
    σ_sq_θ_sq = σ_sq_θ * θ
    μ_0 = np.zeros((2))
    Γ_0 = np.array(
        [
            [σ_sq + σ_sq_θ_sq, σ_sq_θ],
            [σ_sq_θ, σ_sq_θ_sq],
        ]
    )
    return μ_0, Γ_0
