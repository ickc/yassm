"""Define local ratio model using general-linear-gaussian framework,
and then the local ratio model against it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

from yassm.linear_gaussian import LinearGaussianModel
from yassm.linear_gaussian.model.local_ratio import local_ratio_model

if TYPE_CHECKING:
    import numpy.typing as npt


@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def A0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    σ_ε_sq, σ_Z_sq, φ = params
    return np.array([[φ]])


@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def B0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([[1.0]])


# state cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def Σ0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    σ_ε_sq, σ_Z_sq, φ = params
    return np.array([[σ_Z_sq]])


# obs. cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def R0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    σ_ε_sq, σ_Z_sq, φ = params
    return np.array([[σ_ε_sq]])


general_local_ratio_model = LinearGaussianModel(
    3,
    A0,
    B0,
    Σ0,
    R0,
    param_names=local_ratio_model.param_names,
    mle_kwargs={
        "bounds": ((0.0, None), (0.0, None), (0.0, None)),
        "method": "L-BFGS-B",
    },
)
