from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import jit
from numpy.linalg import solve  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from typing import Callable


@dataclass
class GeneralLinearGaussianStateSpaceModel:
    """General linear gaussian state space model.

    :param y: observed data, shape of (n,).
    :param A: function of t returning transition, shape of (d, d).
    :param B: function of t returning design, shape of (n, d).
    :param Σ: function of t returning state covariance, shape of (d, d).
    :param R: function of t returning observation coariance, shape of (n, n).
    :param μ_0: mean of initial X, shape of (d,).
    :param Γ_0: variance of initial X, shape of (d, d).
    :param params0: initial parameters of the model, shape of (N,).

    Y is of n-dimension,
    X is of d-dimension.

        X_0 ~ N(μ_0, Γ_0)
        X_t = A_t X_(t-1) + U_t
        Y_t = B_t X_t + V_t
        U_t ~ N(0, Σ_t)
        V_t ~ N(0, R_t)

        X_s | Y_0 = y_0, ..., Y_t = y_t, θ ~ N(m_s|t, Q_s|t)
    """

    y: np.ndarray
    A: Callable[[np.ndarray, int], np.ndarray]
    B: Callable[[np.ndarray, int], np.ndarray]
    Σ: Callable[[np.ndarray, int], np.ndarray]
    R: Callable[[np.ndarray, int], np.ndarray]
    μ_0: np.ndarray
    Γ_0: np.ndarray
    params0: np.ndarray

    def __post_init__(self):
        y = self.y
        A = self.A
        B = self.B
        Σ = self.Σ
        R = self.R
        μ_0 = self.μ_0
        Γ_0 = self.Γ_0
        params0 = self.params0

        # runtime shape check
        assert y.ndim == 1
        size = y.size
        T = size - 1

        assert μ_0.ndim == 1
        d = μ_0.size
        assert Γ_0.shape == (d, d)

        assert params0.ndim == 1
        N = params0.size

        A0 = A(params0, 0)
        assert A0.shape == (d, d)
        B0 = B(params0, 0)
        assert B0.ndim == 2
        n = B0.shape[0]
        assert B0.shape[1] == d
        Σ0 = Σ(params0, 0)
        assert Σ0.shape == (d, d)
        R0 = R(params0, 0)
        assert R0.shape == (n, n)
        del A0, B0, Σ0, R0

        @jit(
            "Tuple((float64[::1], float64[:, ::1]))(float64[::1], int64, float64[::1], float64[:, ::1])",
            nopython=True,
            nogil=True,
        )
        def one_step_ahead(
            params: np.ndarray,
            s: int,
            m_sMinus1_given_t: np.ndarray,
            Q_sMinus1_given_t: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Calculate one step ahead prediction.

            :param params: paramters of the model, θ.
            :param m_sMinus1_given_t: shape of (d,).
            :param Q_sMinus1_given_t: shape of (d, d).
            :return: (m_s_given_t, Q_s_given_t).
            """
            A_s = A(params, s)
            Σ_s = Σ(params, s)

            m_s_given_t = A_s @ m_sMinus1_given_t
            Q_s_given_t = A_s @ Q_sMinus1_given_t @ A_s.transpose() + Σ_s
            return m_s_given_t, Q_s_given_t

        self.one_step_ahead = one_step_ahead

        @jit(
            "Tuple((float64[::1], float64[:, ::1]))(float64[::1], int64, float64[::1], float64[:, ::1])",
            nopython=True,
            nogil=True,
        )
        def filter_update(
            params: np.ndarray,
            t: int,
            m_t_given_tMinus1: np.ndarray,
            Q_t_given_tMinus1: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Calculate filter update.

            :param params: paramters of the model, θ.
            :param m_t_given_tMinus1: shape of (d,).
            :param Q_t_given_tMinus1: shape of (d, d).
            :return: (m_t_given_t, Q_t_given_t).

            TODO: provide another version using Sherman-Morrison-Woodbury formula
            and dispatch by comparing n, d
            """
            B_t = B(params, t)
            R_t = R(params, t)
            y_t = y[t]

            BQ = B_t @ Q_t_given_tMinus1
            QB = BQ.transpose()
            BQB_plus_R = B_t @ QB + R_t

            m_t_given_t = m_t_given_tMinus1 + QB @ solve(BQB_plus_R, y_t - B_t @ m_t_given_tMinus1)
            Q_t_given_t = Q_t_given_tMinus1 - QB @ solve(BQB_plus_R, BQ)
            return m_t_given_t, Q_t_given_t

        self.filter_update = filter_update

        @jit(
            "Tuple((float64[::1], float64[:, ::1]))(float64[::1], int64, float64[::1], float64[:, ::1], float64[::1],"
            " float64[:, ::1], float64[::1], float64[:, ::1])",
            nopython=True,
            nogil=True,
        )
        def smoothing_backward_recursion(
            params: np.ndarray,
            s: int,
            m_s_given_s: np.ndarray,
            Q_s_given_s: np.ndarray,
            m_sPlus1_given_t: np.ndarray,
            Q_sPlus1_given_t: np.ndarray,
            m_sPlus1_given_s: np.ndarray,
            Q_sPlus1_given_s: np.ndarray,
        ):
            """Calculate filter update.

            :param params: paramters of the model, θ.
            :param m_s_given_s: shape of (d,).
            :param Q_s_given_s: shape of (d, d).
            :param m_sPlus1_given_t: shape of (d,).
            :param Q_sPlus1_given_t: shape of (d, d).
            :param m_sPlus1_given_s: shape of (d,).
            :param Q_sPlus1_given_s: shape of (d, d).
            :return: (m_t_given_t, Q_t_given_t).
            """
            A_sPlus1 = A(params, s + 1)

            Γ_sPlus1_transpose = solve(Q_sPlus1_given_s, A_sPlus1 @ Q_s_given_s)
            Γ_sPlus1 = Γ_sPlus1_transpose.transpose()

            m_s_given_t = m_s_given_s + Γ_sPlus1 @ (m_sPlus1_given_t - m_sPlus1_given_s)
            Q_s_given_t = Q_s_given_s + Γ_sPlus1 @ (Q_sPlus1_given_t - Q_sPlus1_given_s) @ Γ_sPlus1_transpose
            return m_s_given_t, Q_s_given_t

        self.smoothing_backward_recursion = smoothing_backward_recursion

        @jit(
            "Tuple((float64[:, ::1], float64[:, :, ::1], float64[:, ::1], float64[:, :, ::1]))(float64[::1])",
            nopython=True,
            nogil=True,
        )
        def kalman_filtering(params):
            """Kalman filtering.

            :param params: paramters of the model, θ.
            :return: m_t_given_tMinus1, Q_t_given_tMinus1, m_t_given_t, Q_t_given_t
            """
            m_t_given_tMinus1 = np.empty((size, d))
            m_t_given_t = np.empty((size, d))
            Q_t_given_tMinus1 = np.empty((size, d, d))
            Q_t_given_t = np.empty((size, d, d))

            # initialize
            m_t_given_tMinus1[0] = μ_0
            Q_t_given_tMinus1[0] = Γ_0

            t = 0
            temp = filter_update(params, t, m_t_given_tMinus1[t], Q_t_given_tMinus1[t])
            m_t_given_t[t] = temp[0]
            Q_t_given_t[t] = temp[1]
            for t in range(1, size):
                tMinus1 = t - 1
                # s = t, t = t - 1
                temp = one_step_ahead(params, t, m_t_given_t[tMinus1], Q_t_given_t[tMinus1])
                m_t_given_tMinus1[t] = temp[0]
                Q_t_given_tMinus1[t] = temp[1]

                temp = filter_update(params, t, m_t_given_tMinus1[t], Q_t_given_tMinus1[t])
                m_t_given_t[t] = temp[0]
                Q_t_given_t[t] = temp[1]
            return m_t_given_tMinus1, Q_t_given_tMinus1, m_t_given_t, Q_t_given_t

        self.kalman_filtering = kalman_filtering
