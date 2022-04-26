"""General linear Gaussian state space model.

a few functions are jitted here as we want them to be
jitted with the given A, B, Σ, R in order to inline them
or optimized them away.

These cannot be cached as they are compiled using closure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, partial
from itertools import cycle
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import jit, objmode, prange
from scipy import stats

from ..lib import sqrt_diag
from ..lib.linear_gaussian import (
    cov_from_gibbs_samples,
    cov_from_samples,
    filter_update_inplace,
    one_step_ahead_inplace,
    smoothing_backward_recursion_inplace,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
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
    GibbsSampling = Callable[
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
    Likelihoods = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        npt.NDArray[np.float64],
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
class AbstractFit:
    def __post_init__(self) -> None:
        self.y: npt.NDArray[np.float64]
        self.params: npt.NDArray[np.float64]
        self.t: npt.NDArray[Any] | None
        self.param_names: list[str] | None

        self.n: int
        self.size: int

        self.mle_params_chain: npt.NDArray[np.float64]
        raise NotImplementedError

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

    # running functions and cache: _run_*

    def _run_filtered_state_to_obs(self) -> None:
        self._filtered_obs: npt.NDArray[np.float64]
        raise NotImplementedError

    def _run_smoothed_state_to_obs(self) -> None:
        self._smoothed_obs: npt.NDArray[np.float64]
        raise NotImplementedError

    def _run_smoothed_state_to_obs_from_samples(self) -> None:
        logger.info("Running smoothed_state_to_obs from samples...")
        self._smoothed_obs_from_samples: npt.NDArray[np.float64]
        raise NotImplementedError

    def _run_smoothed_state_to_obs_from_gibbs_samples(self) -> None:
        self._smoothed_obs_from_gibbs_samples: npt.NDArray[np.float64]
        raise NotImplementedError

    # cached parameters from _run_*

    @property
    def filtered_obs(self) -> npt.NDArray[np.float64]:
        """Filtered observations of shape (size, n)."""
        if not hasattr(self, "_filtered_obs"):
            self._run_filtered_state_to_obs()
        return self._filtered_obs

    @property
    def smoothed_obs(self) -> npt.NDArray[np.float64]:
        """Smoothed observations of shape (size, n)."""
        if not hasattr(self, "_smoothed_obs"):
            self._run_smoothed_state_to_obs()
        return self._smoothed_obs

    @property
    def smoothed_obs_from_samples(self) -> npt.NDArray[np.float64]:
        """Smoothed observations of shape (size, n) from sampling."""
        if not hasattr(self, "_smoothed_obs_from_samples"):
            self._run_smoothed_state_to_obs_from_samples()
        return self._smoothed_obs_from_samples

    @property
    def smoothed_obs_from_gibbs_samples(self) -> npt.NDArray[np.float64]:
        """Smoothed observations of shape (size, n) from Gibbs sampling."""
        if not hasattr(self, "_smoothed_obs_from_gibbs_samples"):
            self._run_smoothed_state_to_obs_from_gibbs_samples()
        return self._smoothed_obs_from_gibbs_samples

    # CI

    def filtered_obs_CI(
        self,
        α: float,
        with_obs_cov: bool = True,
        n_std: float | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Filtered observations confidence intervals.

        :param α: Confidence level.
        :param with_obs_cov: Whether to include observation covariance.
        :return: tuple of 3 arrays of shape (size, n).
            They are the lower, central and upper confidence intervals.
        """
        raise NotImplementedError

    def smoothed_obs_CI(
        self,
        α: float,
        with_obs_cov: bool = True,
        n_std: float | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Smoothed observations confidence intervals.

        :param α: Confidence level.
        :param with_obs_cov: Whether to include observation covariance.
        :return: tuple of 3 arrays of shape (size, n).
            They are the lower, central and upper confidence intervals.
        """
        raise NotImplementedError

    def smoothed_obs_CI_from_samples(
        self,
        α: float,
        n_std: float | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Smoothed observations confidence intervals from sampling.

        :param α: Confidence level.
        :param with_obs_cov: Whether to include observation covariance.
        :return: tuple of 3 arrays of shape (size, n).
            They are the lower, central and upper confidence intervals.
        """
        raise NotImplementedError

    def smoothed_obs_CI_from_gibbs_samples(
        self,
        α: float,
        n_std: float | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Smoothed observations confidence intervals from Gibbs sampling.

        :param α: Confidence level.
        :param with_obs_cov: Whether to include observation covariance.
        :return: tuple of 3 arrays of shape (size, n).
            They are the lower, central and upper confidence intervals.
        """
        raise NotImplementedError

    def forecast_obs_CI(
        self,
        T_forecast: int,
        α: float,
        n_std: float | None = None,
        with_obs_cov: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Forecast with confidence intervals.

        :param T_forecast: Number of forecast time steps.
        :param α: Confidence level.
        :return: tuple of 3 arrays of shape (size, n).
            They are the lower, central and upper confidence intervals.
        """
        raise NotImplementedError

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
        plot_smoothed_CI_with_obs_cov: bool = True,
        plot_filtered: bool = False,
        plot_filtered_CI: bool = False,
        plot_smoothed: bool = True,
        plot_smoothed_CI: bool = True,
        plot_smoothed_from_samples: bool = False,
        plot_smoothed_from_samples_CI: bool = False,
        plot_smoothed_from_gibbs_samples: bool = False,
        plot_smoothed_from_gibbs_samples_CI: bool = False,
        plot_forecast_CI: bool = True,
        forecast_CI_with_obs_cov: bool = True,
        n_std: float | None = None,
        color: Sequence[tuple[float, float, float]] | None = None,
    ) -> tuple[mpl.figure.Figure, list[mpl.axes._subplots.AxesSubplot]]:
        """Plotting.

        :param α: α-value for C.I.
        :param axes: matplotlib axis, no. of rows should match the dimension of observations.
        :param alpha: transparency level of `ax.fill_between`.
        :param color: a sequence of color of at least 12 elements. Every 2 colors must be pair of light and dark.
            If None, `sns.color_palette("Paired")` is used.

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

        # default colors
        if color is not None and len(color) < 12:
            logger.warning("Too few colors, using default.")
            color = None
        if color is None:
            color = sns.color_palette("Paired")
        # all color pair below are dark-light pairs
        # blue
        colors_obs = (color[1], color[0])
        # green: reserved for forecast
        colors_forecast = (color[3], color[2])
        # these colors should contrasts with blue
        # used for inferences
        # and ordered to avoid red which is close to orange
        colors_iter = cycle(
            iter(
                (
                    # orange
                    (color[7], color[6]),
                    # purple
                    (color[9], color[8]),
                    # brown/yellow
                    (color[11], color[10]),
                    # red
                    (color[5], color[4]),
                )
            )
        )

        str_CI = f" w/ ${(1. - α) * 100.:.3g} \\%%$ C.I." if n_std is None else f" w/ {n_std:.3g}σ C.I."

        colors = colors_obs
        if plot_obs:
            label = "Observation"
            if plot_smoothed_CI_with_obs_cov:
                label += str_CI
            for i, ax in enumerate(axes):
                ax.plot(t, y[:, i], label=label, color=colors[0])

        if plot_smoothed_CI_with_obs_cov:
            smoothed_obs_lower_with_obs_cov, _, smoothed_obs_upper_with_obs_cov = self.smoothed_obs_CI(
                α,
                with_obs_cov=True,
                n_std=n_std,
            )
            for i, ax in enumerate(axes):
                ax.fill_between(
                    t,
                    smoothed_obs_lower_with_obs_cov[:, i],
                    smoothed_obs_upper_with_obs_cov[:, i],
                    alpha=alpha,
                    color=colors[1],
                )

        if plot_filtered or plot_filtered_CI:
            colors = next(colors_iter)
            if plot_filtered:
                filtered_obs = self.filtered_obs
                label = "Filtered obseration"
                if plot_filtered_CI:
                    label += str_CI
                for i, ax in enumerate(axes):
                    ax.plot(t, filtered_obs[:, i], label=label, color=colors[0])

            if plot_filtered_CI:
                filtered_obs_lower, _, filtered_obs_upper = self.filtered_obs_CI(α, with_obs_cov=False, n_std=n_std)
                for i, ax in enumerate(axes):
                    ax.fill_between(
                        t,
                        filtered_obs_lower[:, i],
                        filtered_obs_upper[:, i],
                        alpha=alpha,
                        color=colors[1],
                    )

        if plot_smoothed or plot_smoothed_CI:
            colors = next(colors_iter)
            if plot_smoothed:
                smoothed_obs = self.smoothed_obs
                label = "Smoothed obseration"
                if plot_smoothed_CI:
                    label += str_CI
                for i, ax in enumerate(axes):
                    ax.plot(t, smoothed_obs[:, i], label=label, color=colors[0])

            if plot_smoothed_CI:
                smoothed_obs_lower, _, smoothed_obs_upper = self.smoothed_obs_CI(α, with_obs_cov=False, n_std=n_std)
                for i, ax in enumerate(axes):
                    ax.fill_between(
                        t,
                        smoothed_obs_lower[:, i],
                        smoothed_obs_upper[:, i],
                        alpha=alpha,
                        color=colors[1],
                    )

        if plot_smoothed_from_samples or plot_smoothed_from_samples_CI:
            colors = next(colors_iter)
            if plot_smoothed_from_samples:
                smoothed_obs_from_samples = self.smoothed_obs_from_samples
                label = "Smoothed obseration from samples"
                if plot_smoothed_from_samples_CI:
                    label += str_CI
                for i, ax in enumerate(axes):
                    ax.plot(t, smoothed_obs_from_samples[:, i], label=label, color=colors[0])

            if plot_smoothed_from_samples_CI:
                (
                    smoothed_obs_lower_from_samples,
                    _,
                    smoothed_obs_upper_from_samples,
                ) = self.smoothed_obs_CI_from_samples(α, n_std=n_std)
                for i, ax in enumerate(axes):
                    ax.fill_between(
                        t,
                        smoothed_obs_lower_from_samples[:, i],
                        smoothed_obs_upper_from_samples[:, i],
                        alpha=alpha,
                        color=colors[1],
                    )

        if plot_smoothed_from_gibbs_samples or plot_smoothed_from_gibbs_samples_CI:
            colors = next(colors_iter)
            if plot_smoothed_from_gibbs_samples:
                smoothed_obs_from_gibbs_samples = self.smoothed_obs_from_gibbs_samples
                label = "Smoothed obseration from Gibbs samples"
                if plot_smoothed_from_gibbs_samples_CI:
                    label += str_CI
                for i, ax in enumerate(axes):
                    ax.plot(t, smoothed_obs_from_gibbs_samples[:, i], label=label, color=colors[0])

            if plot_smoothed_from_gibbs_samples_CI:
                (
                    smoothed_obs_lower_from_gibbs_samples,
                    _,
                    smoothed_obs_upper_from_gibbs_samples,
                ) = self.smoothed_obs_CI_from_gibbs_samples(α, n_std=n_std)
                for i, ax in enumerate(axes):
                    ax.fill_between(
                        t,
                        smoothed_obs_lower_from_gibbs_samples[:, i],
                        smoothed_obs_upper_from_gibbs_samples[:, i],
                        alpha=alpha,
                        color=colors[1],
                    )

        if T_forecast > 0:
            label = "Forecast"
            colors = colors_forecast
            t_forecast = self.t_forecast(T_forecast)
            forecast_obs_lower, forecast_obs, forecast_obs_upper = self.forecast_obs_CI(
                T_forecast,
                α,
                n_std=n_std,
                with_obs_cov=forecast_CI_with_obs_cov,
            )
            if plot_forecast_CI:
                label += str_CI
            for i, ax in enumerate(axes):
                ax.plot(t_forecast, forecast_obs[:, i], label=label, color=colors[0])
                if plot_forecast_CI:
                    ax.fill_between(
                        t_forecast,
                        forecast_obs_lower[:, i],
                        forecast_obs_upper[:, i],
                        alpha=alpha,
                        color=colors[1],
                    )

        for i, ax in enumerate(axes):
            ax.legend()
        return fig, axes


@dataclass
class GeneralLinearGaussianStateSpaceModelFit(AbstractFit):
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
    :param gibbs_burn_in: number of burn-in iterations in GibbsSampling algorithm
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
    gibbs_burn_in: int = 0
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

    def _run_filtered_state_to_obs(self) -> None:
        logger.info("Running smoothed_state_to_obs for filtered states...")
        (self._filtered_obs, self._filtered_obs_cov,) = self.parent.smoothed_state_to_obs(
            self.params,
            self.m_t_given_t,
            self.Q_t_given_t,
            0,
            False,
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

    def _run_filtered_state_to_obs_with_obs_cov(self) -> None:
        logger.info("Running smoothed_state_to_obs with obs_cov for filtered states...")
        (self._filtered_obs, self._filtered_obs_cov_with_obs_cov,) = self.parent.smoothed_state_to_obs(
            self.params,
            self.m_t_given_t,
            self.Q_t_given_t,
            0,
            True,
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

    def _run_smoothed_state_to_obs_from_gibbs_samples(self) -> None:
        logger.info("Running smoothed_state_to_obs from Gibbs samples...")
        (
            self._smoothed_obs_from_gibbs_samples,
            self._smoothed_obs_cov_from_gibbs_samples,
        ) = self.parent.smoothed_state_to_obs(
            self.params,
            self.m_t_given_T_from_gibbs_samples,
            self.Q_t_given_T_from_gibbs_samples,
            0,
            False,
        )

    def _run_smoothed_state_to_obs_with_obs_cov_from_gibbs_samples(self) -> None:
        logger.info("Running smoothed_state_to_obs with obs_cov from Gibbs samples...")
        (
            self._smoothed_obs_from_gibbs_samples,
            self._smoothed_obs_cov_with_obs_cov_from_gibbs_samples,
        ) = self.parent.smoothed_state_to_obs(
            self.params,
            self.m_t_given_T_from_gibbs_samples,
            self.Q_t_given_T_from_gibbs_samples,
            0,
            True,
        )

    def _run_gibbs_sampling(self, filter_n_std: float | None = None) -> None:
        """Run Gibbs sampler.

        :param filter_n_std: if not None, the samples are filtered with the given number of standard deviations.
        """
        logger.info("Running gibbs_sampling...")
        self.gibbs_params_chain_with_burn_in, self.gibbs_X_chain_with_burn_in = self.parent.gibbs_sampling(
            self.y,
            self.params,
            self.μ_0,
            self.Γ_0,
            self.n_sample,
        )
        if filter_n_std is not None:
            mask = (
                self.log_likelihoods(self.gibbs_params_chain_with_burn_in)
                > self.log_likelihood - 0.5 * filter_n_std**2
            )
            n_sample_used = self.n_sample - self.gibbs_burn_in
            self.n_sample = mask.sum()
            self.gibbs_burn_in = self.n_sample - n_sample_used
            assert self.gibbs_burn_in >= 0
            self.gibbs_params_chain_with_burn_in = np.ascontiguousarray(self.gibbs_params_chain_with_burn_in[mask])
            self.gibbs_X_chain_with_burn_in = np.ascontiguousarray(self.gibbs_X_chain_with_burn_in[mask])
        self._gibbs_params_chain = self.gibbs_params_chain_with_burn_in[self.gibbs_burn_in :]
        self._gibbs_X_chain = self.gibbs_X_chain_with_burn_in[self.gibbs_burn_in :]

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
    def filtered_obs_cov(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_filtered_obs_cov"):
            self._run_filtered_state_to_obs()
        return self._filtered_obs_cov

    @property
    def smoothed_obs_cov(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_smoothed_obs_cov"):
            self._run_smoothed_state_to_obs()
        return self._smoothed_obs_cov

    @property
    def filtered_obs_cov_with_obs_cov(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_filtered_obs_cov_with_obs_cov"):
            self._run_filtered_state_to_obs_with_obs_cov()
        return self._filtered_obs_cov_with_obs_cov

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
    def smoothed_obs_cov_from_samples(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_smoothed_obs_cov_from_samples"):
            self._run_smoothed_state_to_obs_from_samples()
        return self._smoothed_obs_cov_from_samples

    @property
    def smoothed_obs_cov_with_obs_cov_from_samples(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_smoothed_obs_cov_with_obs_cov_from_samples"):
            self._run_smoothed_state_to_obs_with_obs_cov_from_samples()
        return self._smoothed_obs_cov_with_obs_cov_from_samples

    @property
    def gibbs_params_chain(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_gibbs_params_chain"):
            self._run_gibbs_sampling()
        return self._gibbs_params_chain

    @property
    def gibbs_X_chain(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_gibbs_X_chain"):
            self._run_gibbs_sampling()
        return self._gibbs_X_chain

    @cached_property
    def m_t_given_T_from_gibbs_samples(self) -> npt.NDArray[np.float64]:
        return self.gibbs_X_chain.mean(0)

    @cached_property
    def Q_t_given_T_from_gibbs_samples(self) -> npt.NDArray[np.float64]:
        return cov_from_gibbs_samples(self.gibbs_X_chain)

    @property
    def smoothed_obs_cov_from_gibbs_samples(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_smoothed_obs_cov_from_gibbs_samples"):
            self._run_smoothed_state_to_obs_from_gibbs_samples()
        return self._smoothed_obs_cov_from_gibbs_samples

    @property
    def smoothed_obs_cov_with_obs_cov_from_gibbs_samples(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_smoothed_obs_cov_with_obs_cov_from_gibbs_samples"):
            self._run_smoothed_state_to_obs_with_obs_cov_from_gibbs_samples()
        return self._smoothed_obs_cov_with_obs_cov_from_gibbs_samples

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

    # misc

    def log_likelihoods(
        self,
        params_chain: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return -0.5 * self.parent.neg2_log_likelihoods(params_chain, self.y, self.μ_0, self.Γ_0)

    @cached_property
    def mle_hessian_log_likelihood(self) -> npt.NDArray[np.float64]:
        """Hessian of the log likelihood at MLE."""
        import numdifftools as nd

        logger.info("Computing MLE Hessian, assumes MLE is attained during fit.")
        return -0.5 * nd.Hessian(
            partial(
                self.parent.neg2_log_likelihood,
                y=self.y,
                μ_0=self.μ_0,
                Γ_0=self.Γ_0,
            )
        )(self.params)

    @cached_property
    def posterior_normal_approximation_cov_inv(self) -> npt.NDArray[np.float64]:
        """Posterior inverse covariance matrix of the normal approximation."""
        return -self.mle_hessian_log_likelihood

    @cached_property
    def posterior_normal_approximation_cov(self) -> npt.NDArray[np.float64]:
        """Posterior covariance matrix of the normal approximation."""
        return np.linalg.inv(self.posterior_normal_approximation_cov_inv)

    # CI

    def filtered_obs_CI(
        self,
        α: float,
        with_obs_cov: bool = True,
        n_std: float | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        filtered_obs = self.filtered_obs
        filtered_obs_cov = self.filtered_obs_cov_with_obs_cov if with_obs_cov else self.filtered_obs_cov

        ns_std = stats.norm.ppf([0.5 * α, 1.0 - 0.5 * α]) if n_std is None else (-n_std, n_std)

        filtered_obs_std = sqrt_diag(filtered_obs_cov)
        return (
            filtered_obs + ns_std[0] * filtered_obs_std,
            filtered_obs,
            filtered_obs + ns_std[1] * filtered_obs_std,
        )

    def smoothed_obs_CI(
        self,
        α: float,
        with_obs_cov: bool = True,
        n_std: float | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        smoothed_obs = self.smoothed_obs
        smoothed_obs_cov = self.smoothed_obs_cov_with_obs_cov if with_obs_cov else self.smoothed_obs_cov

        ns_std = stats.norm.ppf([0.5 * α, 1.0 - 0.5 * α]) if n_std is None else (-n_std, n_std)

        smoothed_obs_std = sqrt_diag(smoothed_obs_cov)
        return (
            smoothed_obs + ns_std[0] * smoothed_obs_std,
            smoothed_obs,
            smoothed_obs + ns_std[1] * smoothed_obs_std,
        )

    def smoothed_obs_CI_from_samples(
        self,
        α: float,
        n_std: float | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        smoothed_obs = self.smoothed_obs_from_samples
        if n_std is not None:
            raise NotImplementedError("n_std not implemented")
        CIs = np.quantile(self.samples, [0.5 * α, 1.0 - 0.5 * α], axis=1)
        # cov is redundant here but smoothed_state_to_obs requires it...
        smoothed_obs_low, _ = self.parent.smoothed_state_to_obs(
            self.params,
            CIs[0],
            self.Q_t_given_T,
            0,
            False,
        )
        smoothed_obs_high, _ = self.parent.smoothed_state_to_obs(
            self.params,
            CIs[1],
            self.Q_t_given_T,
            0,
            False,
        )

        return (
            smoothed_obs_low,
            smoothed_obs,
            smoothed_obs_high,
        )

    def smoothed_obs_CI_from_gibbs_samples(
        self,
        α: float,
        n_std: float | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if n_std is not None:
            raise NotImplementedError("n_std not implemented")
        smoothed_obs = self.smoothed_obs_from_gibbs_samples
        CIs = np.quantile(self.gibbs_X_chain, [0.5 * α, 1.0 - 0.5 * α], axis=0)
        # cov is redundant here but smoothed_state_to_obs requires it...
        smoothed_obs_low, _ = self.parent.smoothed_state_to_obs(
            self.params,
            CIs[0],
            self.Q_t_given_T,
            0,
            False,
        )
        smoothed_obs_high, _ = self.parent.smoothed_state_to_obs(
            self.params,
            CIs[1],
            self.Q_t_given_T,
            0,
            False,
        )

        return (
            smoothed_obs_low,
            smoothed_obs,
            smoothed_obs_high,
        )

    def forecast_obs_CI(
        self,
        T_forecast: int,
        α: float,
        n_std: float | None = None,
        with_obs_cov: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        forecast_obs, forecast_obs_cov = self.get_forecast_obs(T_forecast, with_obs_cov=with_obs_cov)

        ns_std = stats.norm.ppf([0.5 * α, 1.0 - 0.5 * α]) if n_std is None else (-n_std, n_std)

        forecast_obs_std = sqrt_diag(forecast_obs_cov)
        return (
            forecast_obs + ns_std[0] * forecast_obs_std,
            forecast_obs,
            forecast_obs + ns_std[1] * forecast_obs_std,
        )


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
    def gibbs_sampling(self) -> GibbsSampling:
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

    @cached_property
    def neg2_log_likelihoods(self) -> Likelihoods:
        """Calculate negative 2 times log-likelihood for param-chain.

        Y|θ
        """
        neg2_log_likelihood = self.neg2_log_likelihood

        @jit(
            "f8[::1](f8[:, ::1], f8[:, ::1], f8[::1], f8[:, ::1])",
            nopython=True,
            nogil=True,
            cache=False,
            parallel=True,
        )
        def neg2_log_likelihoods(
            paramses: npt.NDArray[np.float64],
            y_2d: npt.NDArray[np.float64],
            μ_0_1d: npt.NDArray[np.float64],
            Γ_0_2d: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            """Calculate negative 2 times log-likelihood.

                Y|θ

            This is modified from Kalman filtering but not keeping the intermediate terms.

            This skip the first term that uses the prior of X0 as that is typically a defused distribution.
            """
            N = paramses.shape[0]
            res = np.empty(N)
            for i in prange(N):
                res[i] = neg2_log_likelihood(
                    paramses[i],
                    y_2d,
                    μ_0_1d,
                    Γ_0_2d,
                )
            return res

        return neg2_log_likelihoods

    @property
    def mle_scipy_minimize(self) -> Mle:
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

            opt = minimize(neg2_log_likelihood, params0, args=(y, μ_0, Γ_0), callback=callback, **kwargs)
            msg = opt.message
            if not opt.success:
                raise RuntimeError(f"MLE unccessful with message: {msg}")
            logger.info("MLE constructed with message: %s", msg)
            return opt.x, np.array(chain)

        return mle_scipy_minimize

    def mle(self, method: str = "scipy_minimize") -> Mle:
        """Return the MLE method."""
        method_lower = method.lower()
        if method_lower != "scipy_minimize":
            logger.warning(
                "The only implemented MLE method is scipy_minimize for general linear Gaussian model. Using"
                " scipy_minimize..."
            )
        return self.mle_scipy_minimize

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
        mle_method: str = "",
        mle_n_iter: int = 0,
        mle_return_chain: bool = False,
        mle_kwargs: dict[Any, Any] = {},
        n_sample: int = 0,
        gibbs_burn_in: int = 0,
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
            gibbs_burn_in=gibbs_burn_in,
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
    :param X0: If None, Numba-jit should optimized this away to have no cost, else, a Numba-jitted function of (params) generating μ_0, Γ_0, where the μ_0, Γ_0 passed to `fit` will be ignored.
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
    X0: Callable[[npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] | None = None
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

        if self.X0 is not None:
            logger.info(
                "X0 is provided, μ_0 and Γ_0 are ignored in calculation (but you still need to pass them with valid"
                " type as they will be passed around to functions, albeit not used.)"
            )

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
        X0 = self.X0

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
            if X0 is None:
                m_t_given_tMinus1[0] = μ_0
                Q_t_given_tMinus1[0] = Γ_0
            else:
                μ, Γ = X0(params)
                m_t_given_tMinus1[0] = μ
                Q_t_given_tMinus1[0] = Γ

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

            m_current = m_t_given_t[T]
            Q_T = Q_t_given_t[T]
            # initialize
            with objmode(sample="f8[:, :]"):
                sample = np.random.multivariate_normal(m_current, Q_T, N)
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
                m_current = m_t_given_t[t]
                with objmode(sample="f8[:, :]"):
                    sample = np.random.multivariate_normal(m_current, Q_current, N)
                samples[t] += sample
            return samples

        return backward_sampling

    @cached_property
    def gibbs_sampling(self) -> GibbsSampling:
        kalman_filtering = self.kalman_filtering
        backward_sampling = self.backward_sampling
        sampling_parameters = self.sampling_parameters
        d = self.d

        @jit(
            "Tuple((f8[:, ::1], f8[:, :, ::1]))(f8[:, ::1], f8[::1], f8[::1], f8[:, ::1], i8)",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def gibbs_sampling(
            y: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            μ_0: npt.NDArray[np.float64],
            Γ_0: npt.NDArray[np.float64],
            N: int,
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            """Gibbs sampling.

            :return: the parameters chain and the Xs chain.
            """
            n_params = params.size
            size = y.shape[0]

            # memalloc
            params_chain = np.empty((N, n_params))
            X_chain = np.empty((N, size, d))

            params_cur = params
            for i in range(N):
                (m_t_given_tMinus1, Q_t_given_tMinus1, m_t_given_t, Q_t_given_t,) = kalman_filtering(
                    y,
                    params_cur,
                    μ_0,
                    Γ_0,
                )
                X_chain[i] = backward_sampling(
                    params_cur,
                    m_t_given_tMinus1,
                    Q_t_given_tMinus1,
                    m_t_given_t,
                    Q_t_given_t,
                    1,
                )[:, 0, :]
                params_chain[i] = sampling_parameters(
                    y,
                    X_chain[i],
                )
                params_cur = params_chain[i]
            return params_chain, X_chain

        return gibbs_sampling

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
        X0 = self.X0

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
            if X0 is None:
                m_t_given_tMinus1[:] = μ_0
                Q_t_given_tMinus1[:] = Γ_0
            else:
                μ, Γ = X0(params)
                m_t_given_tMinus1[:] = μ
                Q_t_given_tMinus1[:] = Γ

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
