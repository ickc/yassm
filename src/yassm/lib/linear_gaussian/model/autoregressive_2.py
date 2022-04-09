"""Implements the AR(2) model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit, objmode

if TYPE_CHECKING:
    import numpy.typing as npt

# σ_Z_sq, φ_0, φ_1, φ_2 = params


@jit("f8(f8[::1], f8[::1])", nopython=True, nogil=True)
def neg_log_likelihood(
    params: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> float:
    """Log likelihood of the AR(2) model up to a constant.

    :param params: σ_Z_sq, φ_0, φ_1, φ_2, y_0, y_1.
    :param y: Observations from 0 to T.
        The first 2 elements are unknown
        and should not contains the actual data.
        It will be treated as part of the parameters
        and written over.
    """
    T = y.shape[0] - 1
    σ_Z_sq, φ_0, φ_1, φ_2, y_0, y_1 = params
    y[0] = y_0
    y[1] = y_1
    return 0.5 * T * np.log(σ_Z_sq) + 0.5 / σ_Z_sq * np.square(y[2:] - φ_0 - φ_1 * y[1:-1] - φ_2 * y[0:-2]).sum()


@jit("f8[:, ::1](f8[::1], f8, i8)", nopython=True, nogil=True)
def direct_sampling(
    Y: npt.NDArray[np.float64],
    C: float,
    N: int,
) -> npt.NDArray[np.float64]:
    """Direct sampling from the AR(2) model.

    :param Y: Observations from 2 to T.
    :param C: Y_0, Y_1 ~ Unif(-C, C).
    :param N: Number of samples.

    This one tries to draw Y_0, Y_1 samples too,
    but with a wrong distribution.
    """
    size = Y.size
    T = size + 1

    # memalloc
    X = np.empty((size, 3))
    y_01 = np.empty(2)
    params_chain = np.empty((N, 4))

    # y_0, y_1 to be filled in the sampling below
    t = 0
    X[t, 0] = 1.0
    # X[t, 1] = Y[t - 1]
    # X[t, 2] = Y[t - 2]
    t = 1
    X[t, 0] = 1.0
    X[t, 1] = Y[t - 1]
    # X[t, 2] = Y[t - 2]
    for t in range(2, size):
        X[t, 0] = 1.0
        X[t, 1] = Y[t - 1]
        X[t, 2] = Y[t - 2]

    for i in range(N):
        # we should really be drawing from Y_0, Y_1 | Y_2, ..., Y_T
        # draw Y_0, Y_1 ~ Unif(-C, C)
        y_01 = np.random.uniform(-C, C, 2)
        # y_01 = np.random.normal(0, C, 2)
        X[0, 1] = y_01[1]
        X[0, 2] = y_01[0]
        X[1, 2] = y_01[1]

        X_T_X = X.T @ X
        β_hat = np.linalg.solve(X_T_X, X.T @ Y)

        # draw σ_Z_sq
        σ_Z_sq = 1.0 / np.random.gamma(0.5 * (T - 2), 2.0 / np.square(Y - X @ β_hat).sum())

        params_chain[i, 0] = σ_Z_sq

        with objmode(sample="f8[:, ::1]"):
            sample = np.random.multivariate_normal(β_hat, np.linalg.pinv(X_T_X) * σ_Z_sq, 1)
        params_chain[i, 1:] = sample[0]
    return params_chain


@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True)
def direct_sampling_ignoring_y01(
    y: npt.NDArray[np.float64],
    N: int,
) -> npt.NDArray[np.float64]:
    """Direct sampling from the AR(2) model, ignoring Y_0, Y_1.

    :param Y: Observations from 0 to T.
    :param N: Number of samples.

    This one ignores the unknown y_0, y_1
    and use the first 2 observations as y_0, y_1.
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
    gamma_shape = 0.5 * (T - 2)
    gamma_scale = 2.0 / np.square(Y - X @ β_hat).sum()
    Σ = np.linalg.pinv(X_T_X)
    for i in range(N):
        # draw σ_Z_sq
        σ_Z_sq = 1.0 / np.random.gamma(gamma_shape, gamma_scale)
        params_chain[i, 0] = σ_Z_sq
        with objmode(sample="f8[:, ::1]"):
            sample = np.random.multivariate_normal(β_hat, Σ * σ_Z_sq, 1)
        params_chain[i, 1:] = sample[0]
    return params_chain
