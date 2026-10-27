from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    import numpy.typing as npt

LOG2Π = np.log(2.0 * np.pi)
# R, Σ = σ_ε_sq, σ_Z_sq = params


@jit(
    "Tuple((f8, f8))(f8, f8, f8)",
    nopython=True,
    nogil=True,
    cache=True,
)
def one_step_ahead(
    Σ_s: float,
    m_sMinus1_given_t: float,
    Q_sMinus1_given_t: float,
) -> tuple[float, float]:
    """Calculate one step ahead prediction.

    :return: (m_s_given_t, Q_s_given_t).
    """
    # σ_Z_sq = Σ_s
    m_s_given_t = m_sMinus1_given_t
    Q_s_given_t = Q_sMinus1_given_t + Σ_s
    return m_s_given_t, Q_s_given_t


@jit(
    "Tuple((f8, f8))(f8, f8, f8, f8)",
    nopython=True,
    nogil=True,
    cache=True,
)
def filter_update(
    y_t: float,
    R_t: float,
    m_t_given_tMinus1: float,
    Q_t_given_tMinus1: float,
) -> tuple[float, float]:
    """Calculate filter update.

    :return: (m_t_given_t, Q_t_given_t).
    """
    if np.isnan(y_t):
        return m_t_given_tMinus1, Q_t_given_tMinus1

    # this algorithm uses 3 divisions instead of 1 written in a more straight forward way
    # written this way so that Q = ∞ works
    # but Q or R = 0 wouldn't work now, make sure this wouldn't happen
    σ_ε_sq_inv = 1.0 / R_t
    Q_t_given_tMinus1_inv = 1.0 / Q_t_given_tMinus1

    # this just say they are inverse variance weighted
    Q_t_given_t = 1.0 / (σ_ε_sq_inv + Q_t_given_tMinus1_inv)
    m_t_given_t = (Q_t_given_tMinus1_inv * m_t_given_tMinus1 + σ_ε_sq_inv * y_t) * Q_t_given_t
    return m_t_given_t, Q_t_given_t


@jit(
    "Tuple((f8, f8))(f8, f8, f8, f8, f8)",
    nopython=True,
    nogil=True,
    cache=True,
)
def smoothing_backward_recursion(
    Σ: float,
    m_s_given_s: float,
    Q_s_given_s: float,
    m_sPlus1_given_t: float,
    Q_sPlus1_given_t: float,
) -> tuple[float, float]:
    """Calculate filter update.

    :return: (m_s_given_t, Q_s_given_t).
    """
    # this algorithm uses 3 divisions instead of 1 written in a more straight forward way
    # written this way so that Q = ∞ works
    # but Q or Σ = 0 wouldn't work now, make sure this wouldn't happen
    σ_Z_sq_inv = 1.0 / Σ
    Q_s_given_s_inv = 1.0 / Q_s_given_s

    inv_sum_inv = 1.0 / (σ_Z_sq_inv + Q_s_given_s_inv)

    t2 = σ_Z_sq_inv * inv_sum_inv

    # this just say they are inverse variance weighted
    m_s_given_t = (Q_s_given_s_inv * m_s_given_s + σ_Z_sq_inv * m_sPlus1_given_t) * inv_sum_inv
    Q_s_given_t = inv_sum_inv + t2 * t2 * Q_sPlus1_given_t
    return m_s_given_t, Q_s_given_t


@jit(
    "(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8, f8)",
    nopython=True,
    nogil=True,
    cache=True,
)
def _kalman_filtering_1d_inplace(
    m_t_given_tMinus1: npt.NDArray[np.float64],
    Q_t_given_tMinus1: npt.NDArray[np.float64],
    m_t_given_t: npt.NDArray[np.float64],
    Q_t_given_t: npt.NDArray[np.float64],
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

    # param_names=["obs_cov", "state_cov"]
    # obs_cov
    R_t, Σ_t = params

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
        temp = one_step_ahead(Σ_t, m_t_given_t[tMinus1], Q_t_given_t[tMinus1])
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

    # initialize
    m_t_given_T[T] = m_t_given_t[T]
    Q_t_given_T[T] = Q_t_given_t[T]

    # s = t, t = T
    for s in range(T - 1, -1, -1):
        sPlus1 = s + 1

        temp = smoothing_backward_recursion(
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
    "(f8[:, ::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i8)",
    nopython=True,
    nogil=True,
    cache=True,
)
def _backward_sampling_1d_inplace(
    samples: npt.NDArray[np.float64],
    params: npt.NDArray[np.float64],
    m_t_given_tMinus1: npt.NDArray[np.float64],
    Q_t_given_tMinus1: npt.NDArray[np.float64],
    m_t_given_t: npt.NDArray[np.float64],
    Q_t_given_t: npt.NDArray[np.float64],
    N: int,
) -> None:
    """FFBSampling algorithm.

    :param N: Number of samples to generate.

    Modified from `kalman_smoothing_inplace` as they share
    the same prerequisites.
    """
    size = m_t_given_t.size
    T = size - 1
    Σ = params[1]

    # initialize
    samples[T] = np.random.normal(m_t_given_t[T], np.sqrt(Q_t_given_t[T]), N)

    for t in range(T - 1, -1, -1):
        tPlus1 = t + 1
        Γ_tPlus1 = Q_t_given_t[t] / Q_t_given_tMinus1[tPlus1]
        for i in range(N):
            # this only stores the offset from m_t_given_t[t]
            samples[t, i] = Γ_tPlus1 * (samples[tPlus1, i] - m_t_given_tMinus1[tPlus1])
        # rewrite using definition of Γ and 1-step ahead formula
        # to minimize floating point error
        # the general formula as is would result in negative Q_current
        Q_current = Γ_tPlus1 * Σ
        samples[t] += np.random.normal(m_t_given_t[t], np.sqrt(Q_current), N)


@jit(
    "f8[:, :, ::1](f8[::1], f8[:, ::1], f8[:, :, ::1], f8[:, ::1], f8[:, :, ::1], i8)",
    nopython=True,
    nogil=True,
    cache=True,
)
def backward_sampling(
    params: npt.NDArray[np.float64],
    m_t_given_tMinus1_2d: npt.NDArray[np.float64],
    Q_t_given_tMinus1_3d: npt.NDArray[np.float64],
    m_t_given_t_2d: npt.NDArray[np.float64],
    Q_t_given_t_3d: npt.NDArray[np.float64],
    N: int,
) -> npt.NDArray[np.float64]:
    size = m_t_given_t_2d.shape[0]
    # to 1d
    m_t_given_tMinus1 = m_t_given_tMinus1_2d.flatten()
    Q_t_given_tMinus1 = Q_t_given_tMinus1_3d.flatten()
    m_t_given_t = m_t_given_t_2d.flatten()
    Q_t_given_t = Q_t_given_t_3d.flatten()

    # memalloc
    samples = np.empty((size, N))

    _backward_sampling_1d_inplace(
        samples,
        params,
        m_t_given_tMinus1,
        Q_t_given_tMinus1,
        m_t_given_t,
        Q_t_given_t,
        N,
    )
    return samples.reshape((size, N, 1))


@jit("(f8[::1], f8[::1], f8[::1])", nopython=True, nogil=True, cache=True)
def _sampling_parameters_1d_inplace(
    params: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> None:
    """Sampling parameters given Xs, Ys.

    Step 2 of Gibbs sampling.
    """
    size = y.size
    T = size - 1
    params[0] = 1.0 / np.random.gamma(0.5 * size, 2.0 / np.square(y - x).sum())
    params[1] = 1.0 / np.random.gamma(0.5 * T, 2.0 / np.square(np.diff(x)).sum())


@jit("f8[::1](f8[:, ::1], f8[:, ::1])", nopython=True, nogil=True, cache=True)
def sampling_parameters(
    y_2d: npt.NDArray[np.float64],
    x_2d: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Sampling parameters given Xs, Ys.

    Step 2 of Gibbs sampling.
    """
    y = y_2d.flatten()
    x = x_2d.flatten()
    params = np.empty(2)
    _sampling_parameters_1d_inplace(params, y, x)
    return params


@jit(
    "Tuple((f8[:, ::1], f8[:, :, ::1]))(f8[:, ::1], f8[::1], f8[::1], f8[:, ::1], i8)",
    nopython=True,
    nogil=True,
    cache=True,
)
def gibbs_sampling(
    y_2d: npt.NDArray[np.float64],
    params: npt.NDArray[np.float64],
    μ_0_1d: npt.NDArray[np.float64],
    Γ_0_2d: npt.NDArray[np.float64],
    N: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Gibbs sampling.

    :return: the parameters chain and the Xs chain.
    """
    n_params = params.size
    y = y_2d.flatten()
    μ_0 = μ_0_1d[0]
    Γ_0 = Γ_0_2d[0, 0]
    size = y.size

    # memalloc
    temp = np.empty((4, size))
    m_t_given_tMinus1 = temp[0]
    Q_t_given_tMinus1 = temp[1]
    m_t_given_t = temp[2]
    Q_t_given_t = temp[3]
    params_chain = np.empty((N, n_params))
    X_chain = np.empty((N, size))

    params_cur = params
    for i in range(N):
        _kalman_filtering_1d_inplace(
            m_t_given_tMinus1,
            Q_t_given_tMinus1,
            m_t_given_t,
            Q_t_given_t,
            y,
            params_cur,
            μ_0,
            Γ_0,
        )
        _backward_sampling_1d_inplace(
            X_chain[i].reshape((size, 1)),
            params_cur,
            m_t_given_tMinus1,
            Q_t_given_tMinus1,
            m_t_given_t,
            Q_t_given_t,
            1,
        )
        _sampling_parameters_1d_inplace(
            params_chain[i],
            y,
            X_chain[i],
        )
        params_cur = params_chain[i]
    return params_chain, X_chain.reshape((N, size, 1))


@jit(
    "Tuple((f8[:, ::1], f8[:, :, ::1]))(f8[::1], f8[:, ::1], f8[:, :, ::1], int64, b1)",
    nopython=True,
    nogil=True,
    cache=True,
)
def smoothed_state_to_obs(
    params: npt.NDArray[np.float64],
    m_t: npt.NDArray[np.float64],
    Q_t: npt.NDArray[np.float64],
    t0: int,
    with_obs_cov: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    R_t = params[0]
    return m_t, Q_t + R_t if with_obs_cov else Q_t


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
            Σ_s,
            m_sMinus1_given_T,
            Q_sMinus1_given_T,
        )
        m_t_given_T_predict[s - size] = m_sMinus1_given_T
        Q_t_given_T_predict[s - size] = Q_sMinus1_given_T
    return m_t_given_T_predict.reshape((T_forecast, 1)), Q_t_given_T_predict.reshape((T_forecast, 1, 1))


@jit(
    "f8(f8[::1], f8[:, ::1], f8[::1], f8[:, ::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def neg2_log_likelihood(
    params: npt.NDArray[np.float64],
    y_2d: npt.NDArray[np.float64],
    μ_0_1d: npt.NDArray[np.float64],
    Γ_0_2d: npt.NDArray[np.float64],
) -> float:
    """Calculate negative 2 times log-likelihood.

        Y|θ

    This is modified from Kalman filtering but not keeping the intermediate terms.

    This skip the first term that uses the prior of X0 as that is typically a defused distribution.
    """
    size = y_2d.shape[0]
    T = size - 1
    # to 1d
    y = y_2d.flatten()
    μ_0 = μ_0_1d[0]
    Γ_0 = Γ_0_2d[0, 0]

    # param_names=["obs_cov", "state_cov"]
    # obs_cov
    R_t, Σ_t = params

    # initialize
    m_t_given_tMinus1 = μ_0
    Q_t_given_tMinus1 = Γ_0

    # we don't add t=0 term to res here
    res = T * LOG2Π

    t = 0
    y_t = y[t]
    m_t_given_t, Q_t_given_t = filter_update(y_t, R_t, m_t_given_tMinus1, Q_t_given_tMinus1)
    for t in range(1, size):
        # s = t, t = t - 1
        m_t_given_tMinus1, Q_t_given_tMinus1 = one_step_ahead(Σ_t, m_t_given_t, Q_t_given_t)

        y_t = y[t]
        m_t_given_t, Q_t_given_t = filter_update(y_t, R_t, m_t_given_tMinus1, Q_t_given_tMinus1)

        ε_t = y_t - m_t_given_tMinus1
        H_t = Q_t_given_tMinus1 + R_t
        res += np.log(H_t) + ε_t / H_t * ε_t
    return res


@jit(
    "(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def param_update_inplace_em(
    params: npt.NDArray[np.float64],
    m_t_given_tMinus1: npt.NDArray[np.float64],
    Q_t_given_tMinus1: npt.NDArray[np.float64],
    m_t_given_t: npt.NDArray[np.float64],
    Q_t_given_t: npt.NDArray[np.float64],
    m_t_given_T: npt.NDArray[np.float64],
    Q_t_given_T: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> None:
    size = y.size

    σ_ε_sq = params[0]
    _sum = 0.0
    for s in range(size):
        y_s = y[s]
        Q_s_given_T = Q_t_given_T[s]
        m_s_given_T = m_t_given_T[s]

        # missing at random
        if np.isnan(y_s):
            _sum += σ_ε_sq
        else:
            temp = y_s - m_s_given_T
            _sum += Q_s_given_T + temp * temp
    params[0] = _sum / size

    # if nan is at the beginning of y, and Γ_0 = ∞,
    # then Q_t_given_t[:-1], Q_t_given_tMinus1[1:] can have infinities in the beginning,
    # as the missing at random assumption will propagate the infinity there.
    # in that case there will be a nan and we simply discard it as no information there.
    params[1] = np.nanmean(
        Q_t_given_T[:-1]
        + Q_t_given_T[1:] * (1.0 - 2.0 * (Q_t_given_t[:-1] / Q_t_given_tMinus1[1:]))
        + np.square(m_t_given_T[1:] - m_t_given_T[:-1])
    )
