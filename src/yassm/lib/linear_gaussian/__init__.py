from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit
from numpy.linalg import solve

if TYPE_CHECKING:
    import numpy.typing as npt


@jit(
    "Tuple((f8[::1], f8[:, ::1]))(f8[:, ::1], f8[:, ::1], f8[::1], f8[:, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def one_step_ahead(
    A_s: npt.NDArray[np.float64],
    Σ_s: npt.NDArray[np.float64],
    m_sMinus1_given_t: npt.NDArray[np.float64],
    Q_sMinus1_given_t: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate one step ahead prediction.

    :param m_sMinus1_given_t: shape of (d,).
    :param Q_sMinus1_given_t: shape of (d, d).
    :return: (m_s_given_t, Q_s_given_t).

    TODO: delete this as not used?
    """
    m_s_given_t = A_s @ m_sMinus1_given_t
    Q_s_given_t = A_s @ Q_sMinus1_given_t @ A_s.transpose() + Σ_s
    return m_s_given_t, Q_s_given_t


@jit(
    "(f8[::1], f8[:, ::1], f8[:, ::1], f8[:, ::1], f8[::1], f8[:, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def one_step_ahead_inplace(
    m_s_given_t: npt.NDArray[np.float64],
    Q_s_given_t: npt.NDArray[np.float64],
    A_s: npt.NDArray[np.float64],
    Σ_s: npt.NDArray[np.float64],
    m_sMinus1_given_t: npt.NDArray[np.float64],
    Q_sMinus1_given_t: npt.NDArray[np.float64],
) -> None:
    """Calculate one step ahead prediction.

    :param m_s_given_t: output array to be written.
    :param Q_s_given_t: output array to be written.
    :param m_sMinus1_given_t: shape of (d,).
    :param Q_sMinus1_given_t: shape of (d, d).
    """
    m_s_given_t[:] = A_s @ m_sMinus1_given_t
    Q_s_given_t[:] = A_s @ Q_sMinus1_given_t @ A_s.transpose() + Σ_s


@jit(
    "Tuple((f8[::1], f8[:, ::1]))(f8[::1], f8[:, ::1], f8[:, ::1], f8[::1], f8[:, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def filter_update(
    y_t: npt.NDArray[np.float64],
    B_t: npt.NDArray[np.float64],
    R_t: npt.NDArray[np.float64],
    m_t_given_tMinus1: npt.NDArray[np.float64],
    Q_t_given_tMinus1: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate filter update.

    :param m_t_given_tMinus1: shape of (d,).
    :param Q_t_given_tMinus1: shape of (d, d).
    :return: (m_t_given_t, Q_t_given_t).

    TODO: provide another version using Sherman-Morrison-Woodbury formula
    and dispatch by comparing n, d

    TODO: delete this as not used?
    """
    has_nan = False
    for y_ti in y_t:
        if np.isnan(y_ti):
            has_nan = True
            break
    if has_nan:
        return m_t_given_tMinus1, Q_t_given_tMinus1

    BQ = B_t @ Q_t_given_tMinus1
    QB = BQ.transpose()
    BQB_plus_R = B_t @ QB + R_t

    m_t_given_t = m_t_given_tMinus1 + QB @ solve(BQB_plus_R, y_t - B_t @ m_t_given_tMinus1)
    Q_t_given_t = Q_t_given_tMinus1 - QB @ solve(BQB_plus_R, BQ)
    return m_t_given_t, Q_t_given_t


@jit(
    "(f8[::1], f8[:, ::1], f8[::1], f8[:, ::1], f8[:, ::1], f8[::1], f8[:, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def filter_update_inplace(
    m_t_given_t: npt.NDArray[np.float64],
    Q_t_given_t: npt.NDArray[np.float64],
    y_t: npt.NDArray[np.float64],
    B_t: npt.NDArray[np.float64],
    R_t: npt.NDArray[np.float64],
    m_t_given_tMinus1: npt.NDArray[np.float64],
    Q_t_given_tMinus1: npt.NDArray[np.float64],
) -> None:
    """Calculate filter update.

    :param m_t_given_t: output array to be written.
    :param Q_t_given_t: output array to be written.
    :param m_t_given_tMinus1: shape of (d,).
    :param Q_t_given_tMinus1: shape of (d, d).

    TODO: provide another version using Sherman-Morrison-Woodbury formula
    and dispatch by comparing n, d
    """
    has_nan = False
    for y_ti in y_t:
        if np.isnan(y_ti):
            has_nan = True
            break
    if has_nan:
        m_t_given_t[:] = m_t_given_tMinus1
        Q_t_given_t[:] = Q_t_given_tMinus1
    else:
        BQ = B_t @ Q_t_given_tMinus1
        QB = BQ.transpose()
        BQB_plus_R = B_t @ QB + R_t

        m_t_given_t[:] = m_t_given_tMinus1 + QB @ solve(BQB_plus_R, y_t - B_t @ m_t_given_tMinus1)
        Q_t_given_t[:] = Q_t_given_tMinus1 - QB @ solve(BQB_plus_R, BQ)


@jit(
    "Tuple((f8[::1], f8[:, ::1]))(f8[:, ::1], f8[::1], f8[:, ::1], f8[::1], f8[:, ::1], f8[::1], f8[:, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def smoothing_backward_recursion(
    A_sPlus1: npt.NDArray[np.float64],
    m_s_given_s: npt.NDArray[np.float64],
    Q_s_given_s: npt.NDArray[np.float64],
    m_sPlus1_given_t: npt.NDArray[np.float64],
    Q_sPlus1_given_t: npt.NDArray[np.float64],
    m_sPlus1_given_s: npt.NDArray[np.float64],
    Q_sPlus1_given_s: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate filter update.

    :param m_s_given_s: shape of (d,).
    :param Q_s_given_s: shape of (d, d).
    :param m_sPlus1_given_t: shape of (d,).
    :param Q_sPlus1_given_t: shape of (d, d).
    :param m_sPlus1_given_s: shape of (d,).
    :param Q_sPlus1_given_s: shape of (d, d).
    :return: (m_s_given_t, Q_s_given_t).

    TODO: delete this as not used?
    """
    Γ_sPlus1_transpose = solve(Q_sPlus1_given_s, A_sPlus1 @ Q_s_given_s)
    Γ_sPlus1 = Γ_sPlus1_transpose.transpose()

    m_s_given_t = m_s_given_s + Γ_sPlus1 @ (m_sPlus1_given_t - m_sPlus1_given_s)
    Q_s_given_t = Q_s_given_s + Γ_sPlus1 @ (Q_sPlus1_given_t - Q_sPlus1_given_s) @ Γ_sPlus1_transpose
    return m_s_given_t, Q_s_given_t


@jit(
    "(f8[::1], f8[:, ::1], f8[:, ::1], f8[::1], f8[:, ::1], f8[::1], f8[:, ::1], f8[::1], f8[:, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def smoothing_backward_recursion_inplace(
    m_s_given_t: npt.NDArray[np.float64],
    Q_s_given_t: npt.NDArray[np.float64],
    A_sPlus1: npt.NDArray[np.float64],
    m_s_given_s: npt.NDArray[np.float64],
    Q_s_given_s: npt.NDArray[np.float64],
    m_sPlus1_given_t: npt.NDArray[np.float64],
    Q_sPlus1_given_t: npt.NDArray[np.float64],
    m_sPlus1_given_s: npt.NDArray[np.float64],
    Q_sPlus1_given_s: npt.NDArray[np.float64],
) -> None:
    """Calculate filter update.

    :param m_s_given_t: output array to be written.
    :param Q_s_given_t: output array to be written.
    :param m_s_given_s: shape of (d,).
    :param Q_s_given_s: shape of (d, d).
    :param m_sPlus1_given_t: shape of (d,).
    :param Q_sPlus1_given_t: shape of (d, d).
    :param m_sPlus1_given_s: shape of (d,).
    :param Q_sPlus1_given_s: shape of (d, d).
    """
    Γ_sPlus1_transpose = solve(Q_sPlus1_given_s, A_sPlus1 @ Q_s_given_s)
    Γ_sPlus1 = Γ_sPlus1_transpose.transpose()

    m_s_given_t[:] = m_s_given_s + Γ_sPlus1 @ (m_sPlus1_given_t - m_sPlus1_given_s)
    Q_s_given_t[:] = Q_s_given_s + Γ_sPlus1 @ (Q_sPlus1_given_t - Q_sPlus1_given_s) @ Γ_sPlus1_transpose


@jit(
    "f8[:, :, ::1](f8[:, :, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def cov_from_samples(samples: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    size, N, d = samples.shape
    cov = np.empty((size, d, d))
    for i in range(size):
        cov[i] = np.cov(samples[i], rowvar=False)
    return cov


@jit(
    "f8[:, :, ::1](f8[:, :, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def cov_from_gibbs_samples(samples: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    N, size, d = samples.shape
    cov = np.empty((size, d, d))
    for i in range(size):
        cov[i] = np.cov(samples[:, i], rowvar=False)
    return cov
