from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import jit
from numpy.linalg import solve  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from typing import Callable


@jit(
    "Tuple((f8[::1], f8[:, ::1]))"
    "(f8[:, ::1], f8[:, ::1], f8[::1], f8[:, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def one_step_ahead(
    A_s,
    Σ_s,
    m_sMinus1_given_t: np.ndarray,
    Q_sMinus1_given_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate one step ahead prediction.

    :param m_sMinus1_given_t: shape of (d,).
    :param Q_sMinus1_given_t: shape of (d, d).
    :return: (m_s_given_t, Q_s_given_t).
    """
    m_s_given_t = A_s @ m_sMinus1_given_t
    Q_s_given_t = A_s @ Q_sMinus1_given_t @ A_s.transpose() + Σ_s
    return m_s_given_t, Q_s_given_t


@jit(
    "Tuple((f8[::1], f8[:, ::1]))"
    "(f8[::1], f8[:, ::1], f8[:, ::1], f8[::1], f8[:, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def filter_update(
    y_t: np.ndarray,
    B_t: np.ndarray,
    R_t: np.ndarray,
    m_t_given_tMinus1: np.ndarray,
    Q_t_given_tMinus1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate filter update.

    :param m_t_given_tMinus1: shape of (d,).
    :param Q_t_given_tMinus1: shape of (d, d).
    :return: (m_t_given_t, Q_t_given_t).

    TODO: provide another version using Sherman-Morrison-Woodbury formula
    and dispatch by comparing n, d
    """
    BQ = B_t @ Q_t_given_tMinus1
    QB = BQ.transpose()
    BQB_plus_R = B_t @ QB + R_t

    m_t_given_t = m_t_given_tMinus1 + QB @ solve(BQB_plus_R, y_t - B_t @ m_t_given_tMinus1)
    Q_t_given_t = Q_t_given_tMinus1 - QB @ solve(BQB_plus_R, BQ)
    return m_t_given_t, Q_t_given_t


@jit(
    "Tuple((f8[::1], f8[:, ::1]))"
    "(f8[:, ::1], f8[::1], f8[:, ::1], f8[::1], f8[:, ::1], f8[::1], f8[:, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def smoothing_backward_recursion(
    A_sPlus1: np.ndarray,
    m_s_given_s: np.ndarray,
    Q_s_given_s: np.ndarray,
    m_sPlus1_given_t: np.ndarray,
    Q_sPlus1_given_t: np.ndarray,
    m_sPlus1_given_s: np.ndarray,
    Q_sPlus1_given_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate filter update.

    :param m_s_given_s: shape of (d,).
    :param Q_s_given_s: shape of (d, d).
    :param m_sPlus1_given_t: shape of (d,).
    :param Q_sPlus1_given_t: shape of (d, d).
    :param m_sPlus1_given_s: shape of (d,).
    :param Q_sPlus1_given_s: shape of (d, d).
    :return: (m_t_given_t, Q_t_given_t).
    """
    Γ_sPlus1_transpose = solve(Q_sPlus1_given_s, A_sPlus1 @ Q_s_given_s)
    Γ_sPlus1 = Γ_sPlus1_transpose.transpose()

    m_s_given_t = m_s_given_s + Γ_sPlus1 @ (m_sPlus1_given_t - m_sPlus1_given_s)
    Q_s_given_t = Q_s_given_s + Γ_sPlus1 @ (Q_sPlus1_given_t - Q_sPlus1_given_s) @ Γ_sPlus1_transpose
    return m_s_given_t, Q_s_given_t
