from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

# smoothed_state_to_obs is for export
from .local_level import filter_update, smoothed_state_to_obs

if TYPE_CHECKING:
    import numpy.typing as npt

LOG2Π = np.log(2.0 * np.pi)
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

    Modified from `local_level._backward_sampling_1d_inplace`.
    """
    size = m_t_given_t.size
    T = size - 1
    Σ = params[1]
    φ = params[2]

    # initialize
    samples[T] = np.random.normal(m_t_given_t[T], np.sqrt(Q_t_given_t[T]), N)

    for t in range(T - 1, -1, -1):
        tPlus1 = t + 1
        Γ_tPlus1_over_φ = Q_t_given_t[t] / Q_t_given_tMinus1[tPlus1]
        for i in range(N):
            # this only stores the offset from m_t_given_t[t]
            samples[t, i] = Γ_tPlus1_over_φ * φ * (samples[tPlus1, i] - m_t_given_tMinus1[tPlus1])
        # rewrite using definition of Γ and 1-step ahead formula
        # to minimize floating point error
        # the general formula as is would result in negative Q_current
        Q_current = Γ_tPlus1_over_φ * Σ
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
    """
    Same as local_level.backward_sampling
    """
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

    σ_ε_sq = 1.0 / np.random.gamma(0.5 * size, 2.0 / np.square(y - x).sum(), size=1)[0]
    s_0 = np.square(x[:-1]).sum()
    φ_hat = (x[:-1] * x[1:]).sum() / s_0
    σ_Z_sq = 1.0 / np.random.gamma(0.5 * (T - 1), 2.0 / np.square(φ_hat * x[:-1] - x[1:]).sum(), size=1)[0]
    φ = np.random.normal(φ_hat, np.sqrt(σ_Z_sq / s_0), size=1)[0]

    params[0] = σ_ε_sq
    params[1] = σ_Z_sq
    params[2] = φ


@jit("f8[::1](f8[:, ::1], f8[:, ::1])", nopython=True, nogil=True, cache=True)
def sampling_parameters(
    y_2d: npt.NDArray[np.float64],
    x_2d: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Sampling parameters given Xs, Ys.

    Step 2 of Gibbs sampling.

    Based on `local_level.sampling_parameters`.
    """
    y = y_2d.flatten()
    x = x_2d.flatten()
    params = np.empty(3)
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

    Same as `local_level.gibbs_sampling`.
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

    This is modified from `local_level.neg2_log_likelihood`.

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
    R_t, Σ_t, φ = params

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
        m_t_given_tMinus1, Q_t_given_tMinus1 = one_step_ahead(φ, Σ_t, m_t_given_t, Q_t_given_t)

        y_t = y[t]
        m_t_given_t, Q_t_given_t = filter_update(y_t, R_t, m_t_given_tMinus1, Q_t_given_tMinus1)

        ε_t = y_t - m_t_given_tMinus1
        H_t = Q_t_given_tMinus1 + R_t
        res += np.log(H_t) + ε_t / H_t * ε_t
    return res


@jit(
    "(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def score_inplace(
    out: npt.NDArray[np.float64],
    m_t_given_tMinus1: npt.NDArray[np.float64],
    Q_t_given_tMinus1: npt.NDArray[np.float64],
    m_t_given_t: npt.NDArray[np.float64],
    Q_t_given_t: npt.NDArray[np.float64],
    m_t_given_T: npt.NDArray[np.float64],
    Q_t_given_T: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    params: npt.NDArray[np.float64],
) -> None:
    """Calculate the score vector from data.

    :param out: output parameters of length 3.
    :param temp: holding the intermediate ?_s_given_t stuffs.
    """
    σ_ε_sq, σ_Z_sq, φ = params
    σ_ε = np.sqrt(σ_ε_sq)
    σ_Z = np.sqrt(σ_Z_sq)
    φ_sq = φ * φ

    T_plus1 = y.size
    T = T_plus1 - 1

    # σ_ε_sq
    out[0] = -T_plus1 / σ_ε + 1.0 / (σ_ε * σ_ε_sq) * (Q_t_given_T + np.square(y - m_t_given_T)).sum()
    # σ_Z_sq
    out[1] = (
        -T / σ_Z
        + 1.0
        / (σ_Z_sq * σ_Z)
        * (
            φ_sq * Q_t_given_T[:-1]
            + Q_t_given_T[1:] * (1.0 - 2.0 * φ_sq * (Q_t_given_t[:-1] / Q_t_given_tMinus1[1:]))
            + np.square(m_t_given_T[1:] - φ * m_t_given_T[:-1])
        ).sum()
    )
    # φ
    out[2] = (
        -1.0
        / σ_Z_sq
        * (
            φ * (Q_t_given_T[:-1] - Q_t_given_T[1:] * Q_t_given_t[:-1] / Q_t_given_tMinus1[1:])
            - (m_t_given_T[1:] - φ * m_t_given_T[:-1]) * m_t_given_T[:-1]
        ).sum()
    )


@jit(
    "(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1])",
    nopython=True,
    nogil=True,
    cache=True,
)
def param_update_inplace_gradient(
    params: npt.NDArray[np.float64],
    m_t_given_tMinus1: npt.NDArray[np.float64],
    Q_t_given_tMinus1: npt.NDArray[np.float64],
    m_t_given_t: npt.NDArray[np.float64],
    Q_t_given_t: npt.NDArray[np.float64],
    m_t_given_T: npt.NDArray[np.float64],
    Q_t_given_T: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> None:
    out = np.empty(3)
    score_inplace(
        out,
        m_t_given_tMinus1,
        Q_t_given_tMinus1,
        m_t_given_t,
        Q_t_given_t,
        m_t_given_T,
        Q_t_given_T,
        y,
        params,
    )
    γ = 1e-3
    # if the gradient is negative, reduce to r times the original value
    # in order to not go through the upper half plane
    # r = 0.5
    # for i in range(2):
    #     if out[i] < 0:
    #         γ = min(γ, (r - 1.) * params[i] / out[i])
    params += γ * out


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
    φ_old = params[2]

    # σ_ε_sq
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

    # φ
    params[2] = (
        φ_old * (Q_t_given_T[1:] * Q_t_given_t[:-1] / Q_t_given_tMinus1[1:]).sum()
        + (m_t_given_T[1:] * m_t_given_T[:-1]).sum()
    ) / (Q_t_given_T[:-1] + np.square(m_t_given_T[:-1])).sum()
    φ = params[2]
    φ_sq = φ * φ

    # σ_Z_sq
    # if nan is at the beginning of y, and Γ_0 = ∞,
    # then Q_t_given_t[:-1], Q_t_given_tMinus1[1:] can have infinities in the beginning,
    # as the missing at random assumption will propagate the infinity there.
    # in that case there will be a nan and we simply discard it as no information there.
    params[1] = np.nanmean(
        φ_sq * Q_t_given_T[:-1]
        + Q_t_given_T[1:] * (1.0 - 2.0 * φ * φ_old * (Q_t_given_t[:-1] / Q_t_given_tMinus1[1:]))
        + np.square(m_t_given_T[1:] - φ * m_t_given_T[:-1])
    )
