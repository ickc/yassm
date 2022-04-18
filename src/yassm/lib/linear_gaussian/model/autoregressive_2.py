"""Implements the AR(2) model.

`σ_Z_sq, φ_0, φ_1, φ_2 = params`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit, objmode

if TYPE_CHECKING:
    import numpy.typing as npt


@jit("f8(f8[::1], f8[::1])", nopython=True, nogil=True)
def neg_log_likelihood(
    params: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> float:
    """Log likelihood of the AR(2) model up to a constant.

    :param params: σ_Z_sq, φ_0, φ_1, φ_2, σ_Z_sq must be positive.
    :param y: Observations from 0 to T.
    """
    size = y.size
    T = size + 1
    σ_Z_sq, φ_0, φ_1, φ_2 = params
    return 0.5 * T * np.log(σ_Z_sq) + 0.5 / σ_Z_sq * np.square(y[2:] - φ_0 - φ_1 * y[1:-1] - φ_2 * y[0:-2]).sum()


@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True)
def direct_sampling(
    y: npt.NDArray[np.float64],
    N: int,
) -> npt.NDArray[np.float64]:
    """Direct sampling from the AR(2) model.

    :param Y: Observations from 0 to T.
    :param N: Number of samples.
    """
    Y = y[2:]
    size = Y.size
    T = size + 1

    # memalloc
    X = np.empty((size, 3))
    params_chain = np.empty((N, 4))
    for t in range(size):
        X[t, 0] = 1.0
        X[t, 1] = y[t + 1]
        X[t, 2] = y[t]

    X_T_X = X.T @ X
    β_hat = np.linalg.solve(X_T_X, X.T @ Y)
    gamma_shape = 0.5 * (T - 4)
    gamma_scale = 2.0 / np.square(Y - X @ β_hat).sum()
    Σ = np.linalg.pinv(X_T_X)
    for i in range(N):
        # draw σ_Z_sq
        σ_Z_sq = 1.0 / np.random.gamma(gamma_shape, gamma_scale)
        params_chain[i, 0] = σ_Z_sq
        with objmode(sample="f8[::1]"):
            sample = np.random.multivariate_normal(β_hat, Σ * σ_Z_sq)
        params_chain[i, 1:] = sample
    return params_chain
