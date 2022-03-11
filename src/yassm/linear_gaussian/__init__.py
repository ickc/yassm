from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import jit
from scipy import stats

from ..lib.linear_gaussian import filter_update, one_step_ahead, smoothing_backward_recursion

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable

    import matplotlib as mpl
    import numpy.typing as npt

    KalmanFiltering = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
    ]
    KalmanSmoothing = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
    ]
    SmoothedStateToObs = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
            bool,
        ],
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
    ]
    Forecast = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
        ],
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
    ]
    Mle = Callable[
        [
            npt.NDArray[np.float64],  # y
            npt.NDArray[np.float64],  # params0
            npt.NDArray[np.float64],  # μ_0
            npt.NDArray[np.float64],  # Γ_0
            int,  # n_iter
            bool,  # return_chain
        ],
        tuple[
            npt.NDArray[np.float64],  # params_mle
            ...,  # other custom results such as params_chain
        ],
    ]

logger = getLogger(__name__)


@dataclass
class GeneralLinearGaussianStateSpaceModelFit:
    """A class performing the General linear gaussian state space model pipelines.

    :param parent: a GenericLinearGaussianStateSpaceModel instance.
    :param y: observed data (Y)
    :param params: parameters of the model
    :param μ_0: mean of X0
    :param Γ_0: covariance of X0
    :param mle_n_iter: number of iteration run with MLE
    :param mle_return_chain: if True, return the chain of parameters during MLE
    :param t: optional time axis
    :param param_names: optional names of parameters
    """

    parent: GenericLinearGaussianStateSpaceModel
    _y: npt.ArrayLike
    _params: npt.ArrayLike
    _μ_0: npt.ArrayLike
    _Γ_0: npt.ArrayLike
    mle_n_iter: int = 0
    mle_return_chain: bool = False
    t: npt.NDArray[Any] | None = None
    param_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.y: npt.NDArray[np.float64]
        self.params: npt.NDArray[np.float64]
        self.μ_0: npt.NDArray[np.float64]
        self.Γ_0: npt.NDArray[np.float64]

        self.d: int = self.parent.d
        self.n: int = self.parent.n
        self.N: int = self.parent.N

        d = self.d
        n = self.n
        N = self.N
        t = self.t
        mle_n_iter = self.mle_n_iter
        mle_return_chain = self.mle_return_chain
        param_names = self.param_names

        y = np.ascontiguousarray(self._y, dtype=np.float64)
        params = np.ascontiguousarray(self._params, dtype=np.float64)
        μ_0 = np.ascontiguousarray(self._μ_0, dtype=np.float64)
        Γ_0 = np.ascontiguousarray(self._Γ_0, dtype=np.float64)
        if n == 1 and y.ndim == 1:
            y = y.reshape((-1, 1))
        if d == 1 and Γ_0.ndim == 1:
            Γ_0 = Γ_0.reshape((1, 1))

        self.y = y
        self.params = params
        self.μ_0 = μ_0
        self.Γ_0 = Γ_0

        size, n_p = y.shape
        self.size = size
        assert n_p == n
        del n_p
        assert params.ndim == 1
        assert N == params.size
        assert μ_0.shape == (d,)
        assert Γ_0.shape == (d, d)
        if t is not None:
            assert t.shape == (size,)
        if param_names is None:
            param_names = [f"param_{i}" for i in range(N)]

        if mle_n_iter:
            self.params, self.mle_params_chain = self.parent.mle(y, params, μ_0, Γ_0, mle_n_iter, mle_return_chain)

    def _run_kalman_filtering(self) -> None:
        logger.info("Running kalman filtering...")
        (
            self._m_t_given_tMinus1,
            self._Q_t_given_tMinus1,
            self._m_t_given_t,
            self._Q_t_given_t,
        ) = self.parent.kalman_filtering(
            self.y,
            self.params,
            self.μ_0,
            self.Γ_0,
        )

    def _run_kalman_smoothing(self) -> None:
        logger.info("Running kalman smoothing...")
        (self._m_t_given_T, self._Q_t_given_T,) = self.parent.kalman_smoothing(
            self.y,
            self.params,
            self.m_t_given_tMinus1,
            self.Q_t_given_tMinus1,
            self.m_t_given_t,
            self.Q_t_given_t,
        )

    def _run_smoothed_state_to_obs(self) -> None:
        logger.info("Running smoothed_state_to_obs...")
        (self._smoothed_obs, self._smoothed_obs_cov,) = self.parent.smoothed_state_to_obs(
            self.params,
            self.m_t_given_T,
            self.Q_t_given_T,
            0,
            False,
        )

    def _run_smoothed_state_to_obs_with_obs_cov(self) -> None:
        logger.info("Running smoothed_state_to_obs with obs_cov...")
        (self._smoothed_obs, self._smoothed_obs_cov_with_obs_cov,) = self.parent.smoothed_state_to_obs(
            self.params,
            self.m_t_given_T,
            self.Q_t_given_T,
            0,
            True,
        )

    def get_forecast_state(self, T_forecast: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return self.parent.forecast(
            self.y,
            self.params,
            self.m_t_given_T[-1],
            self.Q_t_given_T[-1],
            T_forecast,
        )

    def get_forecast_obs(
        self,
        T_forecast: int,
        with_obs_cov: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        forecast_state = self.get_forecast_state(T_forecast)
        return self.parent.smoothed_state_to_obs(self.params, *forecast_state, self.size, with_obs_cov)

    @property
    def m_t_given_tMinus1(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_m_t_given_tMinus1"):
            self._run_kalman_filtering()
        return self._m_t_given_tMinus1

    @property
    def Q_t_given_tMinus1(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_Q_t_given_tMinus1"):
            self._run_kalman_filtering()
        return self._Q_t_given_tMinus1

    @property
    def m_t_given_t(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_m_t_given_t"):
            self._run_kalman_filtering()
        return self._m_t_given_t

    @property
    def Q_t_given_t(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_Q_t_given_t"):
            self._run_kalman_filtering()
        return self._Q_t_given_t

    @property
    def m_t_given_T(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_m_t_given_T"):
            self._run_kalman_smoothing()
        return self._m_t_given_T

    @property
    def Q_t_given_T(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_Q_t_given_T"):
            self._run_kalman_smoothing()
        return self._Q_t_given_T

    # alias of statsmodels terminology
    @property
    def filtered_state(self) -> npt.NDArray[np.float64]:
        return self.m_t_given_t

    @property
    def filtered_state_cov(self) -> npt.NDArray[np.float64]:
        return self.Q_t_given_t

    @property
    def predicted_state(self) -> npt.NDArray[np.float64]:
        return self.m_t_given_tMinus1

    @property
    def predicted_state_cov(self) -> npt.NDArray[np.float64]:
        return self.Q_t_given_tMinus1

    @property
    def smoothed_state(self) -> npt.NDArray[np.float64]:
        return self.m_t_given_T

    @property
    def smoothed_state_cov(self) -> npt.NDArray[np.float64]:
        return self.Q_t_given_T

    @property
    def smoothed_obs(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_smoothed_obs"):
            self._run_smoothed_state_to_obs()
        return self._smoothed_obs

    @property
    def smoothed_obs_cov(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_smoothed_obs_cov"):
            self._run_smoothed_state_to_obs()
        return self._smoothed_obs_cov

    @property
    def smoothed_obs_cov_with_obs_cov(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_smoothed_obs_cov_with_obs_cov"):
            self._run_smoothed_state_to_obs_with_obs_cov()
        return self._smoothed_obs_cov_with_obs_cov

    @property
    def mle_params_chain_to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.mle_params_chain, columns=self.param_names)

    @property
    def params_to_series(self) -> pd.Series:
        return pd.Series(self.params, index=self.param_names)

    def plot(
        self,
        T_forecast: int = 0,
        α: float = 0.05,
        axes: list[mpl.axes._subplots.AxesSubplot] | None = None,
        alpha: float = 0.2,
        plot_obs: bool = True,
        plot_smoothed: bool = True,
        plot_smoothed_CI: bool = True,
        plot_smoothed_CI_with_obs_cov: bool = True,
    ) -> tuple[mpl.figure.Figure, list[mpl.axes._subplots.AxesSubplot]]:
        """Plotting.

        :param α: α-value for C.I.
        :param axes: matplotlib axis, no. of rows should match the dimension of observations.
        :param alpha: transparency level of `ax.fill_between`.

        TODO: refactor to a DataFrame first and plot from its data.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        y = self.y
        smoothed_obs = self.smoothed_obs
        size = self.size
        n = self.n

        t_forecast: Iterable[Any]
        if T_forecast > 0:
            if self.t is None:
                t_forecast = range(size, size + T_forecast)
            else:
                dt = self.t[1] - self.t[0]
                start = self.t[-1] + dt
                stop = self.t[-1] + T_forecast * dt
                t_forecast = (
                    pd.date_range(start, stop, T_forecast)
                    if np.issubdtype(self.t.dtype, np.datetime64)
                    else np.linspace(start, stop, T_forecast)
                )
                del dt, start, stop

            forecast_obs, forecast_obs_cov = self.get_forecast_obs(T_forecast)
        t = range(size) if self.t is None else self.t

        ns_std = stats.norm.ppf([0.5 * α, 1.0 - 0.5 * α])

        if axes is None:
            fig, axes = plt.subplots(n)
            if n == 1:
                axes = [axes]
        else:
            fig = plt.gcf()

        color = sns.color_palette("Paired")
        color_obs = color[1]
        color_obs_alt = color[0]
        color_smooth = color[7]
        color_smooth_alt = color[6]
        color_forecast = color[3]
        color_forecast_alt = color[2]
        str_CI = f" w/ ${(1. - α) * 100.:.3g} \\%%$ C.I."

        for i, ax in enumerate(axes):
            s = smoothed_obs[:, i]

            if plot_obs:
                label = "Observation"
                if plot_smoothed_CI_with_obs_cov:
                    label += str_CI
                ax.plot(t, y[:, i], label=label, color=color_obs)
            if plot_smoothed:
                label = "Smoothed obseration"
                if plot_smoothed_CI:
                    label += str_CI

                ax.plot(t, s, label=label, color=color_smooth)

            if plot_smoothed_CI_with_obs_cov:
                smoothed_obs_std = np.sqrt(self.smoothed_obs_cov_with_obs_cov[:, i, i])
                smoothed_obs_lower = s + ns_std[0] * smoothed_obs_std
                smoothed_obs_upper = s + ns_std[1] * smoothed_obs_std
                ax.fill_between(
                    t,
                    smoothed_obs_lower,
                    smoothed_obs_upper,
                    alpha=alpha,
                    color=color_obs_alt,
                )

            if plot_smoothed_CI:
                smoothed_obs_std = np.sqrt(self.smoothed_obs_cov[:, i, i])
                smoothed_obs_lower = s + ns_std[0] * smoothed_obs_std
                smoothed_obs_upper = s + ns_std[1] * smoothed_obs_std
                ax.fill_between(
                    t,
                    smoothed_obs_lower,
                    smoothed_obs_upper,
                    alpha=alpha,
                    color=color_smooth_alt,
                )

            if T_forecast:
                fc = forecast_obs[:, i]
                label = "Forecast" + str_CI
                ax.plot(t_forecast, fc, label=label, color=color_forecast)

                forecast_obs_std = np.sqrt(forecast_obs_cov[:, i, i])
                forecast_obs_lower = fc + ns_std[0] * forecast_obs_std
                forecast_obs_upper = fc + ns_std[1] * forecast_obs_std
                ax.fill_between(
                    t_forecast,
                    forecast_obs_lower,
                    forecast_obs_upper,
                    alpha=alpha,
                    color=color_forecast_alt,
                )

            ax.legend()
        return fig, axes


@dataclass
class GenericLinearGaussianStateSpaceModel:
    """Abstract class for any linear gaussian state space model.

    :param N: no. of params.
    """

    N: int

    def __post_init__(self) -> None:
        # or define this in __init__
        self.param_names: list[str] | None = None

        self.d: int = 0
        self.n: int = 0
        raise NotImplementedError

    @property
    def kalman_filtering(self) -> KalmanFiltering:
        raise NotImplementedError

    @property
    def kalman_smoothing(self) -> KalmanSmoothing:
        raise NotImplementedError

    @property
    def smoothed_state_to_obs(self) -> SmoothedStateToObs:
        raise NotImplementedError

    @property
    def forecast(self) -> Forecast:
        raise NotImplementedError

    @property
    def mle(self) -> Mle:
        raise NotImplementedError

    def fit(
        self,
        y: npt.ArrayLike,
        params: npt.ArrayLike,
        μ_0: npt.ArrayLike,
        Γ_0: npt.ArrayLike,
        mle_n_iter: int = 0,
        mle_return_chain: bool = False,
        t: npt.NDArray[Any] | None = None,
        param_names: list[str] | None = None,
    ) -> GeneralLinearGaussianStateSpaceModelFit:
        """Return a GeneralLinearGaussianStateSpaceModelFit instance.

        :param y: observed data (Y)
        :param params: parameters of the model
        :param μ_0: mean of X0
        :param Γ_0: covariance of X0
        :param mle_n_iter: number of iteration run with MLE
        :param mle_return_chain: if True, return the chain of parameters during MLE
        :param t: optional time axis
        :param param_names: optional names of parameters
        """
        return GeneralLinearGaussianStateSpaceModelFit(
            self,
            y,
            params,
            μ_0,
            Γ_0,
            mle_n_iter=mle_n_iter,
            mle_return_chain=mle_return_chain,
            t=t,
            param_names=param_names or self.param_names,
        )


@dataclass
class GeneralLinearGaussianStateSpaceModel(GenericLinearGaussianStateSpaceModel):
    """General linear gaussian state space model.

    :param N: no. of params.
    :param A0: a Numba0-jitted function of (params) initializing transition, shape of (d, d).
    :param B0: a Numba0-jitted function of (params) initializing design, shape of (n, d).
    :param Σ0: a Numba0-jitted function of (params) initializing state covariance, shape of (d, d).
    :param R0: a Numba0-jitted function of (params) initializing observation coariance, shape of (n, n).
    :param A: If None, Numba-jit should optimized this away to have no cost, else, a Numba0-jitted function of (out, params, t) updating out as transition, shape of (d, d).
    :param B: If None, Numba-jit should optimized this away to have no cost, else, a Numba0-jitted function of (out, params, t) updating out as design, shape of (n, d).
    :param Σ: If None, Numba-jit should optimized this away to have no cost, else, a Numba0-jitted function of (out, params, t) updating out as state covariance, shape of (d, d).
    :param R: If None, Numba-jit should optimized this away to have no cost, else, a Numba0-jitted function of (out, params, t) updating out as observation coariance, shape of (n, n).
    :param param_names: list of names of the parameters.

    Y is of n-dimension,
    X is of d-dimension.

        X_0 ~ N(μ_0, Γ_0)
        X_t = A_t X_(t-1) + U_t
        Y_t = B_t X_t + V_t
        U_t ~ N(0, Σ_t)
        V_t ~ N(0, R_t)

        X_s | Y_0 = y_0, ..., Y_t = y_t, θ ~ N(m_s|t, Q_s|t)
    """

    A0: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    B0: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    Σ0: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    R0: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    A: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int], npt.NDArray[np.float64]] | None = None
    B: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int], npt.NDArray[np.float64]] | None = None
    Σ: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int], npt.NDArray[np.float64]] | None = None
    R: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int], npt.NDArray[np.float64]] | None = None
    param_names: list[str] | None = None

    def __post_init__(self) -> None:
        N = self.N
        A0 = self.A0
        B0 = self.B0
        Σ0 = self.Σ0
        R0 = self.R0

        # runtime shape check
        params = np.zeros(N)
        shape = A0(params).shape
        d = shape[0]
        assert shape == (d, d)
        self.d = d

        shape = B0(params).shape
        assert len(shape) == 2
        assert shape[1] == d
        n = shape[0]
        self.n = n

        assert Σ0(params).shape == (d, d)
        assert R0(params).shape == (n, n)
        del params

    @property
    def kalman_filtering(self) -> KalmanFiltering:
        A0 = self.A0
        B0 = self.B0
        Σ0 = self.Σ0
        R0 = self.R0
        A = self.A
        B = self.B
        Σ = self.Σ
        R = self.R
        d = self.d

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

            # memalloc
            m_t_given_tMinus1 = np.empty((size, d))
            m_t_given_t = np.empty((size, d))
            Q_t_given_tMinus1 = np.empty((size, d, d))
            Q_t_given_t = np.empty((size, d, d))
            A_t = A0(params)
            B_t = B0(params)
            Σ_t = Σ0(params)
            R_t = R0(params)

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

        return kalman_filtering

    @property
    def kalman_smoothing(self) -> KalmanSmoothing:
        A0 = self.A0
        A = self.A
        d = self.d

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
            A_sPlus1 = A0(params)

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

        return kalman_smoothing

    @property
    def smoothed_state_to_obs(self) -> SmoothedStateToObs:
        B0 = self.B0
        R0 = self.R0
        B = self.B
        R = self.R
        n = self.n

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
            size = m_t.shape[0]

            # memalloc
            B_t = B0(params)
            smoothed_obs = np.empty((size, n))
            smoothed_obs_cov = np.empty((size, n, n))
            if with_obs_cov:
                R_t = R0(params)

            for t in range(size):
                if B is not None:
                    B(B_t, params, t0 + t)
                if with_obs_cov and R is not None:
                    R(R_t, params, t0 + t)
                smoothed_obs[t] = B_t @ m_t[t]
                smoothed_obs_cov[t] = B_t @ Q_t[t] @ B_t.transpose()
                if with_obs_cov:
                    smoothed_obs_cov[t] += R_t
            return smoothed_obs, smoothed_obs_cov

        return smoothed_state_to_obs

    @property
    def forecast(self) -> Forecast:
        A0 = self.A0
        Σ0 = self.Σ0
        A = self.A
        Σ = self.Σ
        d = self.d

        @jit(
            "Tuple((f8[:, ::1], f8[:, :, ::1]))(f8[:, ::1], f8[::1], f8[::1], f8[:, ::1], int64)",
            nopython=True,
            nogil=True,
            cache=True,
        )
        def forecast(
            y: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            m_T_given_T: npt.NDArray[np.float64],
            Q_T_given_T: npt.NDArray[np.float64],
            T_forecast: int,
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            size = y.shape[0]
            T = size - 1

            # memalloc
            A_s = A0(params)
            Σ_s = Σ0(params)
            # these indices are offset by size
            # i.e. m_s_given_T_predict = m_t_given_T_predict[s - size]
            m_t_given_T_predict = np.empty((T_forecast, d))
            Q_t_given_T_predict = np.empty((T_forecast, d, d))

            # initialize
            m_sMinus1_given_T = m_T_given_T
            Q_sMinus1_given_T = Q_T_given_T

            for s in range(size, size + T_forecast):
                # t = T
                if A is not None:
                    A(A_s, params, s)
                if Σ is not None:
                    Σ(Σ_s, params, s)
                m_sMinus1_given_T, Q_sMinus1_given_T = one_step_ahead(
                    A_s,
                    Σ_s,
                    m_sMinus1_given_T,
                    Q_sMinus1_given_T,
                )
                m_t_given_T_predict[s - size] = m_sMinus1_given_T
                Q_t_given_T_predict[s - size] = Q_sMinus1_given_T
            return m_t_given_T_predict, Q_t_given_T_predict

        return forecast
