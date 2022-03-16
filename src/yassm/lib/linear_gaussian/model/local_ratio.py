from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

from .local_level import filter_update, smoothed_state_to_obs

if TYPE_CHECKING:
    import numpy.typing as npt

# R, Σ, φ = σ_ε_sq, σ_Z_sq, φ = params


@jit(
    "Tuple((f8, f8))(f8, f8, f8, f8)",
    nopython=True,
    nogil=True,
    cache=True,
)
def one_step_ahead(
    φ: float,
    Σ_s: float,
    m_sMinus1_given_t: float,
    Q_sMinus1_given_t: float,
) -> tuple[float, float]:
    """Calculate one step ahead prediction.

    :return: (m_s_given_t, Q_s_given_t).
    """
    # σ_Z_sq = Σ_s
    m_s_given_t = φ * m_sMinus1_given_t
    Q_s_given_t = φ * φ * Q_sMinus1_given_t + Σ_s
    return m_s_given_t, Q_s_given_t


@jit(
    "Tuple((f8, f8))(f8, f8, f8, f8, f8, f8)",
    nopython=True,
    nogil=True,
    cache=True,
)
def smoothing_backward_recursion(
    φ: float,
    Σ: float,
    m_s_given_s: float,
    Q_s_given_s: float,
    m_sPlus1_given_t: float,
    Q_sPlus1_given_t: float,
) -> tuple[float, float]:
    """Calculate filter update.

    :return: (m_s_given_t, Q_s_given_t).
    """
    σ_Z_sq = Σ
    Q_s_given_s_φ = Q_s_given_s * φ
    Q_s_given_s_φ_sq = Q_s_given_s_φ * φ
    Q_s_given_s_φ_sq_plus_σ_Z_sq_inv = 1.0 / (Q_s_given_s_φ_sq + σ_Z_sq)

    m_s_given_t = (σ_Z_sq * m_s_given_s + Q_s_given_s_φ * m_sPlus1_given_t) * Q_s_given_s_φ_sq_plus_σ_Z_sq_inv
    Q_s_given_t = (
        (σ_Z_sq * σ_Z_sq * Q_s_given_s + (σ_Z_sq + Q_sPlus1_given_t) * Q_s_given_s_φ * Q_s_given_s_φ)
        * Q_s_given_s_φ_sq_plus_σ_Z_sq_inv
        * Q_s_given_s_φ_sq_plus_σ_Z_sq_inv
    )
    return m_s_given_t, Q_s_given_t


@jit(
    "(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8, f8)",
    nopython=True,
    nogil=True,
    cache=True,
)
def _kalman_filtering_1d_inplace(
    m_t_given_tMinus1,
    Q_t_given_tMinus1,
    m_t_given_t,
    Q_t_given_t,
    y: npt.NDArray[np.float64],
    params: npt.NDArray[np.float64],
    μ_0: float,
    Γ_0: float,
) -> None:
    """Kalman filtering.

    :param m_t_given_tMinus1: output array.
    :param Q_t_given_tMinus1: output array.
    :param m_t_given_t: output array.
    :param Q_t_given_t: output array.
    :param params: paramters of the model, θ.
    """
    size = y.shape[0]

    # param_names=["obs_cov", "state_cov", "φ"]
    # obs_cov
    R_t, Σ_t, φ = params

    # initialize
    m_t_given_tMinus1[0] = μ_0
    Q_t_given_tMinus1[0] = Γ_0

    t = 0
    y_t = y[t]
    temp = filter_update(y_t, R_t, m_t_given_tMinus1[t], Q_t_given_tMinus1[t])
    m_t_given_t[t] = temp[0]
    Q_t_given_t[t] = temp[1]
    for t in range(1, size):
        tMinus1 = t - 1
        # s = t, t = t - 1
        temp = one_step_ahead(φ, Σ_t, m_t_given_t[tMinus1], Q_t_given_t[tMinus1])
        m_t_given_tMinus1[t] = temp[0]
        Q_t_given_tMinus1[t] = temp[1]

        y_t = y[t]
        temp = filter_update(y_t, R_t, m_t_given_tMinus1[t], Q_t_given_tMinus1[t])
        m_t_given_t[t] = temp[0]
        Q_t_given_t[t] = temp[1]


@jit(
    "Tuple((f8[:, ::1], f8[:, :, ::1], f8[:, ::1], f8[:, :, ::1]))(f8[:, ::1], f8[::1], f8[::1], f8[:, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def kalman_filtering(
    y_2d: npt.NDArray[np.float64],
    params: npt.NDArray[np.float64],
    μ_0_1d: npt.NDArray[np.float64],
    Γ_0_2d: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Kalman filtering.

    :param params: paramters of the model, θ.
    :return: m_t_given_tMinus1, Q_t_given_tMinus1, m_t_given_t, Q_t_given_t
    """
    size = y_2d.shape[0]
    # to 1d
    y = y_2d.flatten()
    μ_0 = μ_0_1d[0]
    Γ_0 = Γ_0_2d[0, 0]

    # memalloc
    temp = np.empty((4, size))
    m_t_given_tMinus1 = temp[0]
    Q_t_given_tMinus1 = temp[1]
    m_t_given_t = temp[2]
    Q_t_given_t = temp[3]

    _kalman_filtering_1d_inplace(
        m_t_given_tMinus1,
        Q_t_given_tMinus1,
        m_t_given_t,
        Q_t_given_t,
        y,
        params,
        μ_0,
        Γ_0,
    )
    return (
        m_t_given_tMinus1.reshape((size, 1)),
        Q_t_given_tMinus1.reshape((size, 1, 1)),
        m_t_given_t.reshape((size, 1)),
        Q_t_given_t.reshape((size, 1, 1)),
    )


@jit(
    "(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def _kalman_smoothing_1d_inplace(
    m_t_given_T: npt.NDArray[np.float64],
    Q_t_given_T: npt.NDArray[np.float64],
    params: npt.NDArray[np.float64],
    m_t_given_t: npt.NDArray[np.float64],
    Q_t_given_t: npt.NDArray[np.float64],
) -> None:
    size = m_t_given_t.shape[0]
    T = size - 1

    Σ = params[1]
    φ = params[2]

    # initialize
    m_t_given_T[T] = m_t_given_t[T]
    Q_t_given_T[T] = Q_t_given_t[T]

    # s = t, t = T
    for s in range(T - 1, -1, -1):
        sPlus1 = s + 1

        temp = smoothing_backward_recursion(
            φ,
            Σ,
            m_t_given_t[s],
            Q_t_given_t[s],
            m_t_given_T[sPlus1],
            Q_t_given_T[sPlus1],
        )
        m_t_given_T[s] = temp[0]
        Q_t_given_T[s] = temp[1]


@jit(
    "Tuple((f8[:, ::1], f8[:, :, ::1]))(f8[::1], f8[:, ::1], f8[:, :, ::1], f8[:, ::1], f8[:, :, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def kalman_smoothing(
    params: npt.NDArray[np.float64],
    m_t_given_tMinus1_2d: npt.NDArray[np.float64],
    Q_t_given_tMinus1_3d: npt.NDArray[np.float64],
    m_t_given_t_2d: npt.NDArray[np.float64],
    Q_t_given_t_3d: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    size = m_t_given_t_2d.shape[0]
    # to 1d
    m_t_given_t = m_t_given_t_2d.flatten()
    Q_t_given_t = Q_t_given_t_3d.flatten()

    # memalloc
    temp = np.empty((2, size))
    m_t_given_T = temp[0]
    Q_t_given_T = temp[1]

    _kalman_smoothing_1d_inplace(
        m_t_given_T,
        Q_t_given_T,
        params,
        m_t_given_t,
        Q_t_given_t,
    )
    return (
        m_t_given_T.reshape((size, 1)),
        Q_t_given_T.reshape((size, 1, 1)),
    )


@jit(
    "Tuple((f8[:, ::1], f8[:, :, ::1]))(f8[::1], f8[::1], f8[:, ::1], int64)",
    nopython=True,
    nogil=True,
    cache=True,
)
def forecast(
    params: npt.NDArray[np.float64],
    m_T_given_T_1d: npt.NDArray[np.float64],
    Q_T_given_T_2d: npt.NDArray[np.float64],
    T_forecast: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    size = m_T_given_T_1d.shape[0]

    # to 1d
    m_T_given_T = m_T_given_T_1d[0]
    Q_T_given_T = Q_T_given_T_2d[0, 0]

    Σ_s = params[1]
    φ = params[2]

    # memalloc
    # these indices are offset by size
    # i.e. m_s_given_T_predict = m_t_given_T_predict[s - size]
    temp = np.empty((2, T_forecast))
    m_t_given_T_predict = temp[0]
    Q_t_given_T_predict = temp[1]

    # initialize
    m_sMinus1_given_T = m_T_given_T
    Q_sMinus1_given_T = Q_T_given_T

    for s in range(size, size + T_forecast):
        # t = T
        m_sMinus1_given_T, Q_sMinus1_given_T = one_step_ahead(
            φ,
            Σ_s,
            m_sMinus1_given_T,
            Q_sMinus1_given_T,
        )
        m_t_given_T_predict[s - size] = m_sMinus1_given_T
        Q_t_given_T_predict[s - size] = Q_sMinus1_given_T
    return m_t_given_T_predict.reshape((T_forecast, 1)), Q_t_given_T_predict.reshape((T_forecast, 1, 1))
