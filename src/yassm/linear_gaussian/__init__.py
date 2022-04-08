"""General linear Gaussian state space model.

a few functions are jitted here as we want them to be
jitted with the given A, B, Σ, R in order to inline them
or optimized them away.

These cannot be cached as they are compiled using closure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import jit, objmode
from scipy import stats

from ..lib import sqrt_diag
from ..lib.linear_gaussian import (
    cov_from_samples,
    filter_update_inplace,
    one_step_ahead_inplace,
    smoothing_backward_recursion_inplace,
)

if TYPE_CHECKING:
    from typing import Any, Callable

    import matplotlib as mpl
    import numpy.typing as npt
    from typing_extensions import Protocol

    KalmanFilteringInplace = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        None,
    ]
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
    KalmanSmoothingInplace = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        None,
    ]
    KalmanSmoothing = Callable[
        [
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
    FFBSampling = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
        ],
        npt.NDArray[np.float64],
    ]
    SamplingParameters = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        npt.NDArray[np.float64],
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
            int,
        ],
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
    ]
    ParamUpdateInplace = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        None,
    ]
    Likelihood = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        float,
    ]

    class Mle(Protocol):
        def __call__(
            self,
            y: npt.NDArray[np.float64],
            params0: npt.NDArray[np.float64],
            μ_0: npt.NDArray[np.float64],
            Γ_0: npt.NDArray[np.float64],
            n_iter: int,
            return_chain: bool,
            **kwargs: dict[str, Any],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            ...


logger = getLogger(__name__)
LOG2Π = np.log(2.0 * np.pi)


@dataclass
class GeneralLinearGaussianStateSpaceModelFit:
    """A class performing the General linear gaussian state space model pipelines.

    :param parent: a GenericLinearGaussianStateSpaceModel instance.
    :param y: observed data (Y)
    :param params: parameters of the model
    :param μ_0: mean of X0
    :param Γ_0: covariance of X0
    :param mle_method: method used to estimate MLE
    :param mle_n_iter: number of iteration run with MLE
    :param mle_return_chain: if True, return the chain of parameters during MLE
    :param mle_kwargs: pass to self.parent.mle
    :param n_sample: number of samples used in FFBSampling algorithm
    :param t: optional time axis
    :param param_names: optional names of parameters
    """

    parent: GenericLinearGaussianStateSpaceModel
    _y: npt.ArrayLike
    _params: npt.ArrayLike
    _μ_0: npt.ArrayLike
    _Γ_0: npt.ArrayLike
    mle_method: str = "scipy_minimize"
    mle_n_iter: int = 0
    mle_return_chain: bool = False
    mle_kwargs: dict[Any, Any] = field(default_factory=dict)
    n_sample: int = 0
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
            self.params, self.mle_params_chain = self.parent.mle(self.mle_method)(
                y,
                params,
                μ_0,
                Γ_0,
                mle_n_iter,
                mle_return_chain,
                **self.mle_kwargs,
            )

    # defaults

    @property
    def t_array(self) -> npt.ArrayLike:
        return range(self.size) if self.t is None else self.t

    def t_forecast(self, T_forecast: int) -> npt.ArrayLike:
        size = self.size
        if self.t is None:
            return range(size, size + T_forecast)
        else:
            dt = self.t[1] - self.t[0]
            start = self.t[-1] + dt
            stop = self.t[-1] + T_forecast * dt
            return (
                pd.date_range(start, stop, T_forecast)
                if np.issubdtype(self.t.dtype, np.datetime64)
                else np.linspace(start, stop, T_forecast)
            )

    # running functions depending on parameters: get_*

    def get_forecast_state(self, T_forecast: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return self.parent.forecast(
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

    # running functions and cache: _run_*

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

    def _run_smoothed_state_to_obs_from_samples(self) -> None:
        logger.info("Running smoothed_state_to_obs from samples...")
        (self._smoothed_obs_from_samples, self._smoothed_obs_cov_from_samples,) = self.parent.smoothed_state_to_obs(
            self.params,
            self.m_t_given_T_from_samples,
            self.Q_t_given_T_from_samples,
            0,
            False,
        )

    def _run_smoothed_state_to_obs_with_obs_cov_from_samples(self) -> None:
        logger.info("Running smoothed_state_to_obs with obs_cov from samples...")
        (
            self._smoothed_obs_from_samples,
            self._smoothed_obs_cov_with_obs_cov_from_samples,
        ) = self.parent.smoothed_state_to_obs(
            self.params,
            self.m_t_given_T_from_samples,
            self.Q_t_given_T_from_samples,
            0,
            True,
        )

    @cached_property
    def samples(self) -> npt.NDArray[np.float64]:
        logger.info("Running backward_sampling...")
        return self.parent.backward_sampling(
            self.params,
            self.m_t_given_tMinus1,
            self.Q_t_given_tMinus1,
            self.m_t_given_t,
            self.Q_t_given_t,
            self.n_sample,
        )

    # cached parameters from _run_*

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

    @cached_property
    def m_t_given_T_from_samples(self) -> npt.NDArray[np.float64]:
        return self.samples.mean(1)

    @cached_property
    def Q_t_given_T_from_samples(self) -> npt.NDArray[np.float64]:
        return cov_from_samples(self.samples)

    @property
    def smoothed_obs_from_samples(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_smoothed_obs_from_samples"):
            self._run_smoothed_state_to_obs_from_samples()
        return self._smoothed_obs_from_samples

    @property
    def smoothed_obs_cov_from_samples(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_smoothed_obs_cov_from_samples"):
            self._run_smoothed_state_to_obs_from_samples()
        return self._smoothed_obs_cov_from_samples

    @property
    def smoothed_obs_cov_with_obs_cov_from_samples(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_smoothed_obs_cov_with_obs_cov_from_samples"):
            self._run_smoothed_state_to_obs_with_obs_cov_from_samples()
        return self._smoothed_obs_cov_with_obs_cov_from_samples

    # other cached properties

    @cached_property
    def neg2_log_likelihood(self) -> float:
        return self.parent.neg2_log_likelihood(self.params, self.y, self.μ_0, self.Γ_0)

    @property
    def log_likelihood(self) -> float:
        return -0.5 * self.neg2_log_likelihood

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

    # CI

    def smoothed_obs_CI(
        self,
        α: float,
        with_obs_cov: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        smoothed_obs = self.smoothed_obs
        smoothed_obs_cov = self.smoothed_obs_cov_with_obs_cov if with_obs_cov else self.smoothed_obs_cov

        ns_std = stats.norm.ppf([0.5 * α, 1.0 - 0.5 * α])

        smoothed_obs_std = sqrt_diag(smoothed_obs_cov)
        return (
            smoothed_obs + ns_std[0] * smoothed_obs_std,
            smoothed_obs,
            smoothed_obs + ns_std[1] * smoothed_obs_std,
        )

    def forecast_obs_CI(
        self,
        T_forecast: int,
        α: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        forecast_obs, forecast_obs_cov = self.get_forecast_obs(T_forecast)

        ns_std = stats.norm.ppf([0.5 * α, 1.0 - 0.5 * α])

        forecast_obs_std = sqrt_diag(forecast_obs_cov)
        return (
            forecast_obs + ns_std[0] * forecast_obs_std,
            forecast_obs,
            forecast_obs + ns_std[1] * forecast_obs_std,
        )

    # pandas

    @property
    def mle_params_chain_to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.mle_params_chain, columns=self.param_names)

    @property
    def params_to_series(self) -> pd.Series:
        return pd.Series(self.params, index=self.param_names)

    # plot

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
        n = self.n

        t = self.t_array

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

        if plot_obs:
            label = "Observation"
            if plot_smoothed_CI_with_obs_cov:
                label += str_CI
            for i, ax in enumerate(axes):
                ax.plot(t, y[:, i], label=label, color=color_obs)

        if plot_smoothed_CI_with_obs_cov:
            smoothed_obs_lower_with_obs_cov, _, smoothed_obs_upper_with_obs_cov = self.smoothed_obs_CI(
                α,
                with_obs_cov=True,
            )
            for i, ax in enumerate(axes):
                ax.fill_between(
                    t,
                    smoothed_obs_lower_with_obs_cov[:, i],
                    smoothed_obs_upper_with_obs_cov[:, i],
                    alpha=alpha,
                    color=color_obs_alt,
                )

        if plot_smoothed:
            smoothed_obs = self.smoothed_obs
            label = "Smoothed obseration"
            if plot_smoothed_CI:
                label += str_CI
            for i, ax in enumerate(axes):
                ax.plot(t, smoothed_obs[:, i], label=label, color=color_smooth)

        if plot_smoothed_CI:
            smoothed_obs_lower, _, smoothed_obs_upper = self.smoothed_obs_CI(α, with_obs_cov=False)
            for i, ax in enumerate(axes):
                ax.fill_between(
                    t,
                    smoothed_obs_lower[:, i],
                    smoothed_obs_upper[:, i],
                    alpha=alpha,
                    color=color_smooth_alt,
                )

        if T_forecast > 0:
            t_forecast = self.t_forecast(T_forecast)
            forecast_obs_lower, forecast_obs, forecast_obs_upper = self.forecast_obs_CI(T_forecast, α)
            label = "Forecast" + str_CI
            for i, ax in enumerate(axes):
                ax.plot(t_forecast, forecast_obs[:, i], label=label, color=color_forecast)
                ax.fill_between(
                    t_forecast,
                    forecast_obs_lower[:, i],
                    forecast_obs_upper[:, i],
                    alpha=alpha,
                    color=color_forecast_alt,
                )

        for i, ax in enumerate(axes):
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
    def kalman_filtering_inplace(self) -> KalmanFilteringInplace:
        """
        Remember to write your `filter_update_inplace` to handle nan indicating missing at random.
        """
        raise NotImplementedError

    @property
    def kalman_filtering(self) -> KalmanFiltering:
        """
        Remember to write your `filter_update_inplace` to handle nan indicating missing at random.
        """
        raise NotImplementedError

    @property
    def kalman_smoothing_inplace(self) -> KalmanSmoothingInplace:
        raise NotImplementedError

    @property
    def kalman_smoothing(self) -> KalmanSmoothing:
        raise NotImplementedError

    @cached_property
    def backward_sampling(self) -> FFBSampling:
        raise NotImplementedError

    @property
    def sampling_parameters(self) -> SamplingParameters:
        raise NotImplementedError

    @property
    def smoothed_state_to_obs(self) -> SmoothedStateToObs:
        raise NotImplementedError

    @property
    def forecast(self) -> Forecast:
        raise NotImplementedError

    @property
    def neg2_log_likelihood(self) -> Likelihood:
        """Calculate negative 2 times log-likelihood.

        Y|θ
        """
        raise NotImplementedError

    def mle(self, method: str = "gradient") -> Mle:
        raise NotImplementedError

    @staticmethod
    def params_guess(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Estimate initial parameters using the data with law of large number."""
        raise NotImplementedError

    def fit(
        self,
        y: npt.ArrayLike,
        params: npt.ArrayLike | None,
        μ_0: npt.ArrayLike,
        Γ_0: npt.ArrayLike,
        mle_method: str = "scipy_minimize",
        mle_n_iter: int = 0,
        mle_return_chain: bool = False,
        mle_kwargs: dict[Any, Any] = {},
        n_sample: int = 0,
        t: npt.NDArray[Any] | None = None,
        param_names: list[str] | None = None,
    ) -> GeneralLinearGaussianStateSpaceModelFit:
        """Return a GeneralLinearGaussianStateSpaceModelFit instance.

            :param y: observed data (Y)
            :param params: parameters of the model
            :param μ_0: mean of X0
            :param Γ_0: covariance of X0
        :param mle_method: method used to estimate MLE
            :param mle_n_iter: number of iteration run with MLE
            :param mle_return_chain: if True, return the chain of parameters during MLE
            :param t: optional time axis
            :param param_names: optional names of parameters
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        if params is None:
            logger.info("Estimating initial parameters using the data with law of large number...")
            params = self.params_guess(y)
        # using default if exists
        if not mle_kwargs and hasattr(self, "mle_kwargs"):
            mle_kwargs = self.mle_kwargs  # type: ignore[attr-defined]
        return GeneralLinearGaussianStateSpaceModelFit(
            self,
            y,
            params,
            μ_0,
            Γ_0,
            mle_method=mle_method,
            mle_n_iter=mle_n_iter,
            mle_return_chain=mle_return_chain,
            mle_kwargs=mle_kwargs,
            n_sample=n_sample,
            t=t,
            param_names=param_names or self.param_names,
        )


@dataclass
class GeneralLinearGaussianStateSpaceModel(GenericLinearGaussianStateSpaceModel):
    """General linear gaussian state space model.

    :param N: no. of params.
    :param A0: a Numba-jitted function of (params) initializing transition, shape of (d, d).
    :param B0: a Numba-jitted function of (params) initializing design, shape of (n, d).
    :param Σ0: a Numba-jitted function of (params) initializing state covariance, shape of (d, d).
    :param R0: a Numba-jitted function of (params) initializing observation coariance, shape of (n, n).
    :param A: If None, Numba-jit should optimized this away to have no cost, else, a Numba-jitted function of (out, params, t) updating out as transition, shape of (d, d).
    :param B: If None, Numba-jit should optimized this away to have no cost, else, a Numba-jitted function of (out, params, t) updating out as design, shape of (n, d).
    :param Σ: If None, Numba-jit should optimized this away to have no cost, else, a Numba-jitted function of (out, params, t) updating out as state covariance, shape of (d, d).
    :param R: If None, Numba-jit should optimized this away to have no cost, else, a Numba-jitted function of (out, params, t) updating out as observation coariance, shape of (n, n).
    :param B0_forecast: If None, B0 is used for forecasting, Numba-jit should optimized this away to have no cost, else, a Numba-jitted function of (params) initializing design for forecasting, shape of (n, d).
    :param B_forecast: If not None, B0_forecast should be specified. If None, Numba-jit should optimized this away to have no cost, else, a Numba-jitted function of (out, params, t) updating out as design for forecasting, shape of (n, d).
    :param mle_kwargs: pass to self.mle.
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
    B0_forecast: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = None
    B_forecast: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int], npt.NDArray[np.float64]] | None = None
    mle_kwargs: dict[Any, Any] = field(default_factory=dict)
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

    @cached_property
    def kalman_filtering_inplace(self) -> KalmanFilteringInplace:
        A0 = self.A0
        B0 = self.B0
        Σ0 = self.Σ0
        R0 = self.R0
        A = self.A
        B = self.B
        Σ = self.Σ
        R = self.R

        @jit(
            "(f8[:, ::1], f8[:, :, ::1], f8[:, ::1], f8[:, :, ::1], f8[:, ::1], f8[::1], f8[::1], f8[:, ::1])",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def kalman_filtering_inplace(
            m_t_given_tMinus1: npt.NDArray[np.float64],
            Q_t_given_tMinus1: npt.NDArray[np.float64],
            m_t_given_t: npt.NDArray[np.float64],
            Q_t_given_t: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            μ_0: npt.NDArray[np.float64],
            Γ_0: npt.NDArray[np.float64],
        ) -> None:
            """Kalman filtering.

            :param m_t_given_tMinus1: output array.
            :param Q_t_given_tMinus1: output array.
            :param m_t_given_t: output array.
            :param Q_t_given_t: output array.
            :param params: paramters of the model, θ.
            """
            size = y.shape[0]

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
            filter_update_inplace(
                m_t_given_t[t],
                Q_t_given_t[t],
                y_t,
                B_t,
                R_t,
                m_t_given_tMinus1[t],
                Q_t_given_tMinus1[t],
            )
            for t in range(1, size):
                tMinus1 = t - 1
                # s = t, t = t - 1
                if A is not None:
                    A(A_t, params, t)
                if Σ is not None:
                    Σ(Σ_t, params, t)
                one_step_ahead_inplace(
                    m_t_given_tMinus1[t],
                    Q_t_given_tMinus1[t],
                    A_t,
                    Σ_t,
                    m_t_given_t[tMinus1],
                    Q_t_given_t[tMinus1],
                )

                if B is not None:
                    B(B_t, params, t)
                if R is not None:
                    R(R_t, params, t)
                y_t = y[t]
                filter_update_inplace(
                    m_t_given_t[t],
                    Q_t_given_t[t],
                    y_t,
                    B_t,
                    R_t,
                    m_t_given_tMinus1[t],
                    Q_t_given_tMinus1[t],
                )

        return kalman_filtering_inplace

    @cached_property
    def kalman_filtering(self) -> KalmanFiltering:
        d = self.d
        kalman_filtering_inplace = self.kalman_filtering_inplace

        @jit(
            "Tuple((f8[:, ::1], f8[:, :, ::1], f8[:, ::1], f8[:, :, ::1]))(f8[:, ::1], f8[::1], f8[::1], f8[:, ::1])",
            nopython=True,
            nogil=True,
            cache=False,
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
            temp = np.empty(2 * size * d * (d + 1))
            m_t_given_tMinus1 = temp[: size * d].reshape((size, d))
            Q_t_given_tMinus1 = temp[size * d : size * d * (d + 1)].reshape((size, d, d))
            m_t_given_t = temp[size * d * (d + 1) : size * d * (d + 2)].reshape((size, d))
            Q_t_given_t = temp[size * d * (d + 2) :].reshape((size, d, d))

            kalman_filtering_inplace(
                m_t_given_tMinus1,
                Q_t_given_tMinus1,
                m_t_given_t,
                Q_t_given_t,
                y,
                params,
                μ_0,
                Γ_0,
            )
            return m_t_given_tMinus1, Q_t_given_tMinus1, m_t_given_t, Q_t_given_t

        return kalman_filtering

    @cached_property
    def kalman_smoothing_inplace(self) -> KalmanSmoothingInplace:
        A0 = self.A0
        A = self.A

        @jit(
            "(f8[:, ::1], f8[:, :, ::1], f8[::1], f8[:, ::1], f8[:, :, ::1], f8[:, ::1], f8[:, :, ::1])",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def kalman_smoothing_inplace(
            m_t_given_T: npt.NDArray[np.float64],
            Q_t_given_T: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            m_t_given_tMinus1: npt.NDArray[np.float64],
            Q_t_given_tMinus1: npt.NDArray[np.float64],
            m_t_given_t: npt.NDArray[np.float64],
            Q_t_given_t: npt.NDArray[np.float64],
        ) -> None:
            size = m_t_given_t.shape[0]
            T = size - 1

            # memalloc
            A_sPlus1 = A0(params)

            # initialize
            m_t_given_T[T] = m_t_given_t[T]
            Q_t_given_T[T] = Q_t_given_t[T]

            # s = t, t = T
            for s in range(T - 1, -1, -1):
                sPlus1 = s + 1
                if A is not None:
                    A(A_sPlus1, params, sPlus1)

                smoothing_backward_recursion_inplace(
                    m_t_given_T[s],
                    Q_t_given_T[s],
                    A_sPlus1,
                    m_t_given_t[s],
                    Q_t_given_t[s],
                    m_t_given_T[sPlus1],
                    Q_t_given_T[sPlus1],
                    m_t_given_tMinus1[sPlus1],
                    Q_t_given_tMinus1[sPlus1],
                )

        return kalman_smoothing_inplace

    @cached_property
    def kalman_smoothing(self) -> KalmanSmoothing:
        kalman_smoothing_inplace = self.kalman_smoothing_inplace
        d = self.d

        @jit(
            "Tuple((f8[:, ::1], f8[:, :, ::1]))(f8[::1], f8[:, ::1], f8[:, :, ::1], f8[:, ::1], f8[:, :, ::1])",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def kalman_smoothing(
            params: npt.NDArray[np.float64],
            m_t_given_tMinus1: npt.NDArray[np.float64],
            Q_t_given_tMinus1: npt.NDArray[np.float64],
            m_t_given_t: npt.NDArray[np.float64],
            Q_t_given_t: npt.NDArray[np.float64],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            size = m_t_given_t.shape[0]

            # memalloc
            temp = np.empty(size * d * (d + 1))
            m_t_given_T = temp[: size * d].reshape((size, d))
            Q_t_given_T = temp[size * d : size * d * (d + 1)].reshape((size, d, d))

            kalman_smoothing_inplace(
                m_t_given_T,
                Q_t_given_T,
                params,
                m_t_given_tMinus1,
                Q_t_given_tMinus1,
                m_t_given_t,
                Q_t_given_t,
            )
            return m_t_given_T, Q_t_given_T

        return kalman_smoothing

    @cached_property
    def backward_sampling(self) -> FFBSampling:
        A0 = self.A0
        A = self.A

        @jit(
            "f8[:, :, ::1](f8[::1], f8[:, ::1], f8[:, :, ::1], f8[:, ::1], f8[:, :, ::1], i8)",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def backward_sampling(
            params: npt.NDArray[np.float64],
            m_t_given_tMinus1: npt.NDArray[np.float64],
            Q_t_given_tMinus1: npt.NDArray[np.float64],
            m_t_given_t: npt.NDArray[np.float64],
            Q_t_given_t: npt.NDArray[np.float64],
            N: int,
        ) -> npt.NDArray[np.float64]:
            """FFBSampling algorithm.

            :param N: Number of samples to generate.

            Modified from `kalman_smoothing_inplace` as they share
            the same prerequisites.
            """
            size, d = m_t_given_t.shape
            T = size - 1

            # memalloc
            A_tPlus1 = A0(params)
            samples = np.empty((size, N, d))
            Γ_tPlus1_transpose = np.empty((d, d))
            Q_current = np.empty((d, d))

            # initialize
            with objmode(sample="f8[:, :]"):
                sample = np.random.multivariate_normal(m_t_given_t[T], Q_t_given_t[T], N)
            samples[T] = sample

            for t in range(T - 1, -1, -1):
                tPlus1 = t + 1
                if A is not None:
                    A(A_tPlus1, params, tPlus1)
                Γ_tPlus1_transpose[:] = np.linalg.solve(Q_t_given_tMinus1[tPlus1], A_tPlus1 @ Q_t_given_t[t])
                for i in range(N):
                    # this only stores the offset from m_t_given_t[t]
                    samples[t, i] = Γ_tPlus1_transpose.transpose() @ (samples[tPlus1, i] - m_t_given_tMinus1[tPlus1])
                Q_current[:] = (
                    Q_t_given_t[t] - Γ_tPlus1_transpose.transpose() @ Q_t_given_tMinus1[tPlus1] @ Γ_tPlus1_transpose
                )
                with objmode(sample="f8[:, :]"):
                    sample = np.random.multivariate_normal(m_t_given_t[t], Q_current, N)
                samples[t] += sample
            return samples

        return backward_sampling

    @cached_property
    def smoothed_state_to_obs(self) -> SmoothedStateToObs:
        B0 = self.B0
        B = self.B
        B0_forecast = self.B0_forecast
        B_forecast = self.B_forecast
        R0 = self.R0
        R = self.R
        n = self.n

        @jit(
            "Tuple((f8[:, ::1], f8[:, :, ::1]))(f8[::1], f8[:, ::1], f8[:, :, ::1], int64, b1)",
            nopython=True,
            nogil=True,
            cache=False,
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
            B_t_forecast = B_t if B0_forecast is None else B0_forecast(params)
            temp = np.empty(size * n * (n + 1))
            smoothed_obs = temp[: size * n].reshape((size, n))
            smoothed_obs_cov = temp[size * n : size * n * (n + 1)].reshape((size, n, n))
            if with_obs_cov:
                R_t = R0(params)

            for t in range(size):
                if B is not None:
                    B(B_t, params, t0 + t)
                if B0_forecast is not None and B_forecast is not None:
                    B_forecast(B_t_forecast, params, t0 + t)
                if with_obs_cov and R is not None:
                    R(R_t, params, t0 + t)
                smoothed_obs[t] = B_t_forecast @ m_t[t]
                if with_obs_cov:
                    smoothed_obs_cov[t] = B_t @ Q_t[t] @ B_t.transpose() + R_t
                else:
                    smoothed_obs_cov[t] = B_t_forecast @ Q_t[t] @ B_t_forecast.transpose()
            return smoothed_obs, smoothed_obs_cov

        return smoothed_state_to_obs

    @cached_property
    def forecast(self) -> Forecast:
        A0 = self.A0
        Σ0 = self.Σ0
        A = self.A
        Σ = self.Σ
        d = self.d

        @jit(
            "Tuple((f8[:, ::1], f8[:, :, ::1]))(f8[::1], f8[::1], f8[:, ::1], int64)",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def forecast(
            params: npt.NDArray[np.float64],
            m_T_given_T: npt.NDArray[np.float64],
            Q_T_given_T: npt.NDArray[np.float64],
            T_forecast: int,
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            size = m_T_given_T.shape[0]

            # memalloc
            A_s = A0(params)
            Σ_s = Σ0(params)
            # these indices are offset by size
            # i.e. m_s_given_T_predict = m_t_given_T_predict[s - size]
            temp = np.empty(T_forecast * d * (d + 1))
            m_t_given_T_predict = temp[: T_forecast * d].reshape((T_forecast, d))
            Q_t_given_T_predict = temp[T_forecast * d : T_forecast * d * (d + 1)].reshape((T_forecast, d, d))

            # initialize
            m_sMinus1_given_T = m_T_given_T
            Q_sMinus1_given_T = Q_T_given_T

            for s in range(size, size + T_forecast):
                # t = T
                if A is not None:
                    A(A_s, params, s)
                if Σ is not None:
                    Σ(Σ_s, params, s)
                one_step_ahead_inplace(
                    m_t_given_T_predict[s - size],
                    Q_t_given_T_predict[s - size],
                    A_s,
                    Σ_s,
                    m_sMinus1_given_T,
                    Q_sMinus1_given_T,
                )
                m_sMinus1_given_T = m_t_given_T_predict[s - size]
                Q_sMinus1_given_T = Q_t_given_T_predict[s - size]
            return m_t_given_T_predict, Q_t_given_T_predict

        return forecast

    @cached_property
    def neg2_log_likelihood(self) -> Likelihood:
        """Calculate negative 2 times log-likelihood.

        Y|θ
        """
        A0 = self.A0
        B0 = self.B0
        Σ0 = self.Σ0
        R0 = self.R0
        A = self.A
        B = self.B
        Σ = self.Σ
        R = self.R
        d = self.d
        n = self.n

        @jit(
            "f8(f8[::1], f8[:, ::1], f8[::1], f8[:, ::1])",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def neg2_log_likelihood(
            params: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            μ_0: npt.NDArray[np.float64],
            Γ_0: npt.NDArray[np.float64],
        ) -> float:
            """Calculate negative 2 times log-likelihood.

                Y|θ

            This is modified from Kalman filtering but not keeping the intermediate terms.

            This skip the first term that uses the prior of X0 as that is typically a defused distribution.
            """
            size = y.shape[0]
            T = size - 1

            A_t = A0(params)
            B_t = B0(params)
            Σ_t = Σ0(params)
            R_t = R0(params)

            # memalloc
            # this only holds the value at step-t
            temp = np.empty(2 * d * (d + 1) + n * (n + 1))
            m_t_given_tMinus1 = temp[:d]
            Q_t_given_tMinus1 = temp[d : d * (d + 1)].reshape((d, d))
            m_t_given_t = temp[d * (d + 1) : d * (d + 2)]
            Q_t_given_t = temp[d * (d + 2) : d * (2 * d + 2)].reshape((d, d))
            ε_t = temp[d * (2 * d + 2) : d * (2 * d + 2) + n]
            H_t = temp[d * (2 * d + 2) + n :].reshape((n, n))

            # initialize
            m_t_given_tMinus1[:] = μ_0
            Q_t_given_tMinus1[:] = Γ_0

            # we don't add t=0 term to res here
            res = T * LOG2Π

            t = 0
            if B is not None:
                B(B_t, params, 0)
            if R is not None:
                R(R_t, params, 0)
            y_t = y[t]
            filter_update_inplace(
                m_t_given_t,
                Q_t_given_t,
                y_t,
                B_t,
                R_t,
                m_t_given_tMinus1,
                Q_t_given_tMinus1,
            )
            for t in range(1, size):
                # s = t, t = t - 1
                if A is not None:
                    A(A_t, params, t)
                if Σ is not None:
                    Σ(Σ_t, params, t)
                one_step_ahead_inplace(
                    m_t_given_tMinus1,
                    Q_t_given_tMinus1,
                    A_t,
                    Σ_t,
                    m_t_given_t,
                    Q_t_given_t,
                )

                if B is not None:
                    B(B_t, params, t)
                if R is not None:
                    R(R_t, params, t)
                y_t = y[t]
                filter_update_inplace(
                    m_t_given_t,
                    Q_t_given_t,
                    y_t,
                    B_t,
                    R_t,
                    m_t_given_tMinus1,
                    Q_t_given_tMinus1,
                )

                ε_t[:] = y_t - B_t @ m_t_given_tMinus1
                H_t[:] = B_t @ Q_t_given_tMinus1 @ B_t.transpose() + R_t
                res += np.log(np.linalg.det(H_t)) + ε_t @ np.linalg.solve(H_t, ε_t)
            return res

        return neg2_log_likelihood

    def mle(self, method: str = "scipy_minimize") -> Mle:
        method_lower = method.lower()
        if method_lower != "scipy_minimize":
            logger.warning(
                "The only implemented MLE method is scipy_minimize for general linear Gaussian model. Using"
                " scipy_minimize..."
            )
        neg2_log_likelihood = self.neg2_log_likelihood

        def mle_scipy_minimize(
            y: npt.NDArray[np.float64],
            params0: npt.NDArray[np.float64],
            μ_0: npt.NDArray[np.float64],
            Γ_0: npt.NDArray[np.float64],
            n_iter: int,
            return_chain: bool,
            **kwargs: dict[str, Any],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            """Calculate MLE from log-likelihood using scipy.optimize.minimize.

            Tips: include `bounds` for constrainted variables.
            """
            from scipy.optimize import minimize

            def _callback(xk: npt.NDArray[np.float64]) -> None:
                """Store iterations."""
                chain.append(xk)

            if n_iter:
                logger.warning("n_iter %s is ignored when using scipy.optimize.minimize.", n_iter)

            chain: list[npt.NDArray[np.float64]] = []
            callack: Callable[[npt.NDArray[np.float64]], None] | None
            if return_chain:
                chain.append(params0)
                callback = _callback
            else:
                callback = None

            opt = minimize(neg2_log_likelihood, params0, args=(y.reshape(-1, 1), μ_0, Γ_0), callback=callback, **kwargs)
            msg = opt.message
            if not opt.success:
                raise RuntimeError(f"MLE unccessful with message: {msg}")
            logger.info("MLE constructed with message: %s", msg)
            return opt.x, np.array(chain)

        return mle_scipy_minimize
