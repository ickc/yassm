from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
from numba import jit

from ..lib.linear_gaussian import filter_update, one_step_ahead, smoothing_backward_recursion

if TYPE_CHECKING:
    from typing import Callable

    import numpy.typing as npt

logger = getLogger(__name__)


@dataclass
class GeneralLinearGaussianStateSpaceModelFit:
    """A class performing the General linear gaussian state space model pipelines."""

    y: npt.NDArray[np.float64]
    params: npt.NDArray[np.float64]
    μ_0: npt.NDArray[np.float64]
    Γ_0: npt.NDArray[np.float64]
    kalman_filtering: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ]
    kalman_smoothing: Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ]
    d: int
    n: int
    t: npt.NDArray | None = None
    param_names: list[str] | None = None

    def __post_init__(self) -> None:
        y = self.y
        params = self.params
        μ_0 = self.μ_0
        Γ_0 = self.Γ_0
        d = self.d
        n = self.n
        t = self.t
        param_names = self.param_names

        size, np = y.shape
        assert np == n
        assert params.ndim == 1
        N = params.size
        assert μ_0.shape == (d,)
        assert Γ_0.shape == (d, d)
        if t is not None:
            assert t.shape == (size,)
        if param_names is None:
            param_names = [f"param_{i}" for i in range(N)]

    def _run_kalman_filtering(self):
        logger.info("Running kalman filtering...")
        (
            self._m_t_given_tMinus1,
            self._Q_t_given_tMinus1,
            self._m_t_given_t,
            self._Q_t_given_t,
        ) = self.kalman_filtering(
            self.y,
            self.params,
            self.μ_0,
            self.Γ_0,
        )

    def _run_kalman_smoothing(self):
        logger.info("Running kalman smoothing...")
        (self._m_t_given_T, self._Q_t_given_T,) = self.kalman_smoothing(
            self.y,
            self.params,
            self.m_t_given_tMinus1,
            self.Q_t_given_tMinus1,
            self.m_t_given_t,
            self.Q_t_given_t,
        )

    @property
    def m_t_given_tMinus1(self):
        if not hasattr(self, "_m_t_given_tMinus1"):
            self._run_kalman_filtering()
        return self._m_t_given_tMinus1

    @property
    def Q_t_given_tMinus1(self):
        if not hasattr(self, "_Q_t_given_tMinus1"):
            self._run_kalman_filtering()
        return self._Q_t_given_tMinus1

    @property
    def m_t_given_t(self):
        if not hasattr(self, "_m_t_given_t"):
            self._run_kalman_filtering()
        return self._m_t_given_t

    @property
    def Q_t_given_t(self):
        if not hasattr(self, "_Q_t_given_t"):
            self._run_kalman_filtering()
        return self._Q_t_given_t

    @property
    def m_t_given_T(self):
        if not hasattr(self, "_m_t_given_T"):
            self._run_kalman_smoothing()
        return self._m_t_given_T

    @property
    def Q_t_given_T(self):
        if not hasattr(self, "_Q_t_given_T"):
            self._run_kalman_smoothing()
        return self._Q_t_given_T


@dataclass
class GeneralLinearGaussianStateSpaceModel:
    """General linear gaussian state space model.


    :param A0: initial transition, shape of (d, d).
    :param B0: initial design, shape of (n, d).
    :param Σ0: initial state covariance, shape of (d, d).
    :param R0: initial observation coariance, shape of (n, n).
    :param A: If None, Numba-jit should optimized this away to have no cost, else, a Numba0-jitted function of (out, params, t) updating out as transition, shape of (d, d).
    :param B: If None, Numba-jit should optimized this away to have no cost, else, a Numba0-jitted function of (out, params, t) updating out as design, shape of (n, d).
    :param Σ: If None, Numba-jit should optimized this away to have no cost, else, a Numba0-jitted function of (out, params, t) updating out as state covariance, shape of (d, d).
    :param R: If None, Numba-jit should optimized this away to have no cost, else, a Numba0-jitted function of (out, params, t) updating out as observation coariance, shape of (n, n).

    Y is of n-dimension,
    X is of d-dimension.

        X_0 ~ N(μ_0, Γ_0)
        X_t = A_t X_(t-1) + U_t
        Y_t = B_t X_t + V_t
        U_t ~ N(0, Σ_t)
        V_t ~ N(0, R_t)

        X_s | Y_0 = y_0, ..., Y_t = y_t, θ ~ N(m_s|t, Q_s|t)
    """

    A0: npt.NDArray[np.float64]
    B0: npt.NDArray[np.float64]
    Σ0: npt.NDArray[np.float64]
    R0: npt.NDArray[np.float64]
    A: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int]] | None
    B: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int]] | None
    Σ: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int]] | None
    R: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int]] | None

    def __post_init__(self) -> None:
        A0 = self.A0
        B0 = self.B0
        Σ0 = self.Σ0
        R0 = self.R0
        A = self.A
        B = self.B
        Σ = self.Σ
        R = self.R

        # runtime shape check
        d = A0.shape[0]
        self.d = d
        assert A0.shape == (d, d)
        assert B0.ndim == 2
        n = B0.shape[0]
        self.n = n
        assert B0.shape[1] == d
        assert Σ0.shape == (d, d)
        assert R0.shape == (n, n)

        @jit(
            "Tuple((f8[:, ::1], f8[:, :, ::1], f8[:, ::1], f8[:, :, ::1]))(f8[:, ::1], f8[::1], f8[::1], f8[:, ::1])",
            nopython=True,
            nogil=True,
            cache=True,
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

            # memalloc
            m_t_given_tMinus1 = np.empty((size, d))
            m_t_given_t = np.empty((size, d))
            Q_t_given_tMinus1 = np.empty((size, d, d))
            Q_t_given_t = np.empty((size, d, d))
            A_t = A0.copy()
            B_t = B0.copy()
            Σ_t = Σ0.copy()
            R_t = R0.copy()

            # initialize
            m_t_given_tMinus1[0] = μ_0
            Q_t_given_tMinus1[0] = Γ_0

            t = 0
            if B is not None:
                B(B_t, params, 0)
            if R is not None:
                R(R_t, params, 0)
            y_t = y[t]
            temp = filter_update(y_t, B_t, R_t, m_t_given_tMinus1[t], Q_t_given_tMinus1[t])
            m_t_given_t[t] = temp[0]
            Q_t_given_t[t] = temp[1]
            for t in range(1, size):
                tMinus1 = t - 1
                # s = t, t = t - 1
                if A is not None:
                    A(A_t, params, t)
                if Σ is not None:
                    Σ(Σ_t, params, t)
                temp = one_step_ahead(A_t, Σ_t, m_t_given_t[tMinus1], Q_t_given_t[tMinus1])
                m_t_given_tMinus1[t] = temp[0]
                Q_t_given_tMinus1[t] = temp[1]

                if B is not None:
                    B(B_t, params, t)
                if R is not None:
                    R(R_t, params, t)
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
            cache=True,
        )
        def kalman_smoothing(
            y: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            m_t_given_tMinus1: npt.NDArray[np.float64],
            Q_t_given_tMinus1: npt.NDArray[np.float64],
            m_t_given_t: npt.NDArray[np.float64],
            Q_t_given_t: npt.NDArray[np.float64],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            size = y.shape[0]
            T = size - 1

            # memalloc
            m_t_given_T = np.empty((size, d))
            Q_t_given_T = np.empty((size, d, d))
            A_sPlus1 = A0.copy()

            # initialize
            m_t_given_T[T] = m_t_given_t[T]
            Q_t_given_T[T] = Q_t_given_t[T]

            # s = t, t = T
            for s in range(T - 1, -1, -1):
                sPlus1 = s + 1
                if A is not None:
                    A(A_sPlus1, params, sPlus1)

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

    def fit(
        self,
        y: npt.NDArray[np.float64],
        params: npt.NDArray[np.float64],
        μ_0: npt.NDArray[np.float64],
        Γ_0: npt.NDArray[np.float64],
        t: npt.NDArray | None = None,
        param_names: list[str] | None = None,
    ) -> GeneralLinearGaussianStateSpaceModelFit:
        return GeneralLinearGaussianStateSpaceModelFit(
            y,
            params,
            μ_0,
            Γ_0,
            self.kalman_filtering,
            self.kalman_smoothing,
            self.d,
            self.n,
            t=t,
            param_names=param_names,
        )
