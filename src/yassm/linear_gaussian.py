from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import jit

from .lib.linear_gaussian import filter_update, one_step_ahead, smoothing_backward_recursion

if TYPE_CHECKING:
    from typing import Callable

    import numpy.typing as npt


@dataclass
class GeneralLinearGaussianStateSpaceModel:
    """General linear gaussian state space model.

    :param A: function of (params, t) returning transition, shape of (d, d).
    :param B: function of (params, t) returning design, shape of (n, d).
    :param Σ: function of (params, t) returning state covariance, shape of (d, d).
    :param R: function of (params, t) returning observation coariance, shape of (n, n).
    :param N: total number of parameters.

    Y is of n-dimension,
    X is of d-dimension.

        X_0 ~ N(μ_0, Γ_0)
        X_t = A_t X_(t-1) + U_t
        Y_t = B_t X_t + V_t
        U_t ~ N(0, Σ_t)
        V_t ~ N(0, R_t)

        X_s | Y_0 = y_0, ..., Y_t = y_t, θ ~ N(m_s|t, Q_s|t)
    """

    A: Callable[[npt.NDArray[np.float64], int], npt.NDArray[np.float64]]
    B: Callable[[npt.NDArray[np.float64], int], npt.NDArray[np.float64]]
    Σ: Callable[[npt.NDArray[np.float64], int], npt.NDArray[np.float64]]
    R: Callable[[npt.NDArray[np.float64], int], npt.NDArray[np.float64]]
    N: int

    def __post_init__(self) -> None:
        A = self.A
        B = self.B
        Σ = self.Σ
        R = self.R
        N = self.N

        # runtime shape check
        params = np.zeros(N)
        A0 = A(params, 0)
        d = A0.shape[0]
        assert A0.shape == (d, d)
        B0 = B(params, 0)
        assert B0.ndim == 2
        n = B0.shape[0]
        assert B0.shape[1] == d
        Σ0 = Σ(params, 0)
        assert Σ0.shape == (d, d)
        R0 = R(params, 0)
        assert R0.shape == (n, n)
        del A0, B0, Σ0, R0, params

        @jit(
            "Tuple((f8[:, ::1], f8[:, :, ::1], f8[:, ::1], f8[:, :, ::1]))(f8[:, ::1], f8[::1], f8[::1], f8[:, ::1])",
            nopython=True,
            nogil=True,
        )
        def kalman_filtering(
            y: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            μ_0: npt.NDArray[np.float64],
            Γ_0: npt.NDArray[np.float64],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            """Kalman filtering.

            :param params: paramters of the model, θ.
            :return: m_t_given_tMinus1, Q_t_given_tMinus1, m_t_given_t, Q_t_given_t
            """
            size = y.shape[0]
            T = size - 1

            m_t_given_tMinus1 = np.empty((size, d))
            m_t_given_t = np.empty((size, d))
            Q_t_given_tMinus1 = np.empty((size, d, d))
            Q_t_given_t = np.empty((size, d, d))

            # initialize
            m_t_given_tMinus1[0] = μ_0
            Q_t_given_tMinus1[0] = Γ_0

            t = 0
            B_t = B(params, t)
            R_t = R(params, t)
            y_t = y[t]
            temp = filter_update(y_t, B_t, R_t, m_t_given_tMinus1[t], Q_t_given_tMinus1[t])
            m_t_given_t[t] = temp[0]
            Q_t_given_t[t] = temp[1]
            for t in range(1, size):
                tMinus1 = t - 1
                # s = t, t = t - 1
                A_s = A(params, t)
                Σ_s = Σ(params, t)
                temp = one_step_ahead(A_s, Σ_s, m_t_given_t[tMinus1], Q_t_given_t[tMinus1])
                m_t_given_tMinus1[t] = temp[0]
                Q_t_given_tMinus1[t] = temp[1]

                B_t = B(params, t)
                R_t = R(params, t)
                y_t = y[t]
                temp = filter_update(y_t, B_t, R_t, m_t_given_tMinus1[t], Q_t_given_tMinus1[t])
                m_t_given_t[t] = temp[0]
                Q_t_given_t[t] = temp[1]
            return m_t_given_tMinus1, Q_t_given_tMinus1, m_t_given_t, Q_t_given_t

        self.kalman_filtering = kalman_filtering

        @jit(
            "Tuple((f8[:, ::1], f8[:, :, ::1]))"
            "(f8[:, ::1], f8[::1], f8[:, ::1], f8[:, :, ::1], f8[:, ::1], f8[:, :, ::1])",
            nopython=True,
            nogil=True,
        )
        def kalman_smoothing(
            y: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            m_t_given_tMinus1,
            Q_t_given_tMinus1,
            m_t_given_t,
            Q_t_given_t,
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            size = y.shape[0]
            T = size - 1

            m_t_given_T = np.empty((size, d))
            Q_t_given_T = np.empty((size, d, d))

            # initialize
            m_t_given_T[T] = m_t_given_t[T]
            Q_t_given_T[T] = Q_t_given_t[T]

            # s = t, t = T
            for s in range(T - 1, -1, -1):
                sPlus1 = s + 1
                A_sPlus1 = A(params, sPlus1)

                temp = smoothing_backward_recursion(
                    A_sPlus1,
                    m_t_given_t[s],
                    Q_t_given_t[s],
                    m_t_given_T[sPlus1],
                    Q_t_given_T[sPlus1],
                    m_t_given_tMinus1[sPlus1],
                    Q_t_given_tMinus1[sPlus1],
                )
                m_t_given_T[s] = temp[0]
                Q_t_given_T[s] = temp[1]
            return m_t_given_T, Q_t_given_T

        self.kalman_smoothing = kalman_smoothing
