from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit, prange, vectorize
from numpy.linalg import solve

if TYPE_CHECKING:
    import numpy.typing as npt

SQRT_2Π_INV = 1.0 / np.sqrt(2.0 * np.pi)

# probaility functions #########################################################


@vectorize("f8(f8, f8, f8, b1)", nopython=True)
def prob_cauchy(
    x: npt.NDArray[np.float64],
    x_0: float,
    τ_sq: float,
    norm_peak_to_1: bool,
) -> npt.NDArray[np.float64]:
    """Probability of a Cauchy distribution."""
    return (
        τ_sq / (τ_sq + np.square((x - x_0)))
        if norm_peak_to_1
        else (np.sqrt(τ_sq) / np.pi) / (τ_sq + np.square((x - x_0)))
    )


@vectorize("f8(f8, f8, f8, b1)", nopython=True)
def prob_normal_1d(
    x: npt.NDArray[np.float64],
    μ: float,
    cov_inv: float,
    norm_peak_to_1: bool,
) -> npt.NDArray[np.float64]:
    """Probability density of normal distribution.

    :param cov_inv: inverse covariance.
    :param norm_peak_to_1: Normalize peak to 1.
    """
    scale = 1.0 if norm_peak_to_1 else SQRT_2Π_INV * np.sqrt(cov_inv)
    return scale * np.exp(-0.5 * cov_inv * np.square(x - μ))


@jit("f8(f8[::1], f8[::1], f8[:, ::1], b1)", nopython=True, nogil=True, cache=True)
def prob_normal_each(
    x: npt.NDArray[np.float64],
    μ: npt.NDArray[np.float64],
    Σ_inv: npt.NDArray[np.float64],
    norm_peak_to_1: bool,
) -> npt.NDArray[np.float64]:
    """Probability density of normal distribution.

    :param Σ_inv: Inverse of covariance matrix.
    :param norm_peak_to_1: Normalize peak to 1.
    """
    Σ_inv_neg_half = -0.5 * Σ_inv
    scale = 1.0 if norm_peak_to_1 else SQRT_2Π_INV * np.sqrt(np.linalg.det(Σ_inv))
    d = x - μ
    return np.exp(d @ Σ_inv_neg_half @ d) * scale


@jit("f8[::1](f8[:, ::1], f8[::1], f8[:, ::1], b1)", nopython=True, nogil=True, cache=True, parallel=True)
def prob_normal(
    x: npt.NDArray[np.float64],
    μ: npt.NDArray[np.float64],
    Σ_inv: npt.NDArray[np.float64],
    norm_peak_to_1: bool,
) -> npt.NDArray[np.float64]:
    """Probability density of normal distribution.

    :param Σ_inv: Inverse of covariance matrix.
    :param norm_peak_to_1: Normalize peak to 1.
    """
    n = x.shape[0]
    res = np.empty(n)
    Σ_inv_neg_half = -0.5 * Σ_inv
    scale = 1.0 if norm_peak_to_1 else SQRT_2Π_INV * np.sqrt(np.linalg.det(Σ_inv))
    for i in prange(n):
        d = x[i] - μ
        res[i] = np.exp(d @ Σ_inv_neg_half @ d) * scale
    return res


# pipelines ####################################################################


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
