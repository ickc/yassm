"""General state space model using particle filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
from numba import get_num_threads, jit, objmode, prange

from ..lib import random_choice_weighted_inplace, sqrt_diag
from ..lib.linear_gaussian import cov_from_samples, prob_normal_each
from ..linear_gaussian import AbstractFit, GeneralLinearGaussianStateSpaceModel, GenericLinearGaussianStateSpaceModel

if TYPE_CHECKING:
    from typing import Any, Callable

    import numpy.typing as npt
    from typing_extensions import Protocol

    SamplingX0Inplace = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        None,
    ]
    SamplingX0TildeInplace = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        None,
    ]
    SamplingOneStepAheadInplace = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
        ],
        None,
    ]
    SamplingQInplace = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
        ],
        None,
    ]
    WeightsInplace = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
        ],
        None,
    ]
    ParticleFilter = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
        ],
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
    ]
    LikelihoodFromWeights = Callable[
        [npt.NDArray[np.float64]],
        float,
    ]
    Likelihood = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
        ],
        float,
    ]
    Likelihoods = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
        ],
        npt.NDArray[np.float64],
    ]
    SmoothedStateToObs = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
        ],
        npt.NDArray[np.float64],
    ]
    Forecast = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int,
        ],
        npt.NDArray[np.float64],
    ]
    TransformFunc = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

    class Mle(Protocol):
        def __call__(
            self,
            y: npt.NDArray[np.float64],
            params0: npt.NDArray[np.float64],
            n_iter: int,
            n_sample: int,
            return_chain: bool,
            **kwargs: dict[str, Any],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            ...


logger = getLogger(__name__)


@dataclass
class GenericParticleFilterStateSpaceModelFit(AbstractFit):
    """A class performing the particle filter state space model pipelines.

    :param parent: a GenericParticleFilterStateSpaceModel instance.
    :param y: observed data (Y)
    :param params: parameters of the model
    :param mle_method: method used to estimate MLE
    :param mle_n_iter: number of iteration run with MLE
    :param mle_return_chain: if True, return the chain of parameters during MLE
    :param mle_kwargs: pass to self.parent.mle
    :param n_sample: number of samples used in FFBSampling algorithm
    :param t: optional time axis
    :param param_names: optional names of parameters

    TODO: tailor for particle filter
    """

    parent: GenericParticleFilterStateSpaceModel
    _y: npt.ArrayLike
    _params: npt.ArrayLike
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

        self.d: int = self.parent.d
        self.n: int = self.parent.n
        self.N: int = self.parent.N

        d = self.d
        n = self.n
        N = self.N
        t = self.t
        mle_n_iter = self.mle_n_iter
        mle_return_chain = self.mle_return_chain
        n_sample = self.n_sample
        param_names = self.param_names

        y = np.ascontiguousarray(self._y, dtype=np.float64)
        params = np.ascontiguousarray(self._params, dtype=np.float64)
        if n == 1 and y.ndim == 1:
            y = y.reshape((-1, 1))

        self.y = y
        self.params = params

        size, n_p = y.shape
        self.size = size
        assert n_p == n
        del n_p
        assert params.ndim == 1
        assert N == params.size
        if t is not None:
            assert t.shape == (size,)
        if param_names is None:
            param_names = [f"param_{i}" for i in range(N)]

        if mle_n_iter:
            logger.info("Using %s for MLE.", self.mle_method)
            logger.warning("n_iter is ignored.")
            if self.mle_method == "grid_search":
                params_grid = self.mle_kwargs["params_grid"]
                ls = self.parent.neg_log_likelihoods(params_grid, y, n_sample)
                idx = np.argmin(ls)
                self.params = params_grid[idx]
                self.neg_log_likelihoods_grid = ls
            elif self.mle_method == "scipy_minimize":
                self.params, self.mle_params_chain = self.parent.mle_scipy_minimize(
                    y,
                    params,
                    mle_n_iter,
                    n_sample,
                    mle_return_chain,
                    **self.mle_kwargs,
                )

    # running functions depending on parameters: get_*

    def get_forecast_state(self, T_forecast: int) -> npt.NDArray[np.float64]:
        forecast: Forecast = self.parent.forecast
        return forecast(
            self.params,
            self.x_t,
            T_forecast,
        )

    def get_forecast_obs(
        self,
        T_forecast: int,
        with_obs_cov: bool = True,
    ) -> npt.NDArray[np.float64]:
        if with_obs_cov:
            raise NotImplementedError("with_obs_cov is not implemented yet.")
        forecast_state = self.get_forecast_state(T_forecast)
        return self.parent.smoothed_state_to_obs(self.params, forecast_state, self.size)

    # running functions and cache: _run_*

    def _run_filtered_state_to_obs(self) -> None:
        filtered_obs_samples = self.filtered_obs_samples
        self._filtered_obs: npt.NDArray[np.float64] = filtered_obs_samples.mean(axis=1)

    def _run_particle_filter(self) -> None:
        """Run the particle filter.

        :return: None
        """
        logger.info("Running particle_filter")
        self._x_t_tilde, self._x_t, self._w_t = self.parent.particle_filter(self.y, self.params, self.n_sample)

    # cached parameters from _run_*

    @property
    def x_t_tilde(self) -> npt.NDArray[np.float64]:
        """x_t_tilde of shape (size, N, d)."""
        if not hasattr(self, "_x_t_tilde"):
            self._run_particle_filter()
        return self._x_t_tilde

    @property
    def x_t(self) -> npt.NDArray[np.float64]:
        """x_t of shape (size, N, d)."""
        if not hasattr(self, "_x_t"):
            self._run_particle_filter()
        return self._x_t

    @property
    def w_t(self) -> npt.NDArray[np.float64]:
        """w_t of shape (size, N)."""
        if not hasattr(self, "_w_t"):
            self._run_particle_filter()
        return self._w_t

    # Gassian-like descriptions

    @cached_property
    def m_t_given_tMinus1(self) -> npt.NDArray[np.float64]:
        """m_t_given_tMinus1 of shape (size, d)."""
        if self.parent.particle_filter_method != "bootstrap":
            raise NotImplementedError("m_t_given_tMinus1 is only implemented for bootstrap")
        return self.x_t_tilde.mean(axis=1)

    @cached_property
    def Q_t_given_tMinus1(self) -> npt.NDArray[np.float64]:
        """Q_t_given_tMinus1 of shape (size, d, d)."""
        if self.parent.particle_filter_method != "bootstrap":
            raise NotImplementedError("Q_t_given_tMinus1 is only implemented for bootstrap")
        return cov_from_samples(self.x_t_tilde)

    @cached_property
    def m_t_given_t(self) -> npt.NDArray[np.float64]:
        """m_t_given_t of shape (size, d)."""
        return self.x_t.mean(axis=1)

    @cached_property
    def Q_t_given_t(self) -> npt.NDArray[np.float64]:
        """Q_t_given_t of shape (size, d)."""
        return cov_from_samples(self.x_t)

    @cached_property
    def filtered_obs_samples(self) -> npt.NDArray[np.float64]:
        """Return all samples of filtered states in observation space.

        :return: filtered_obs_samples of shape (size, N, n)
        """
        return self.parent.smoothed_state_to_obs(self.params, self.x_t, 0)

    @cached_property
    def m_t_given_t_obs(self) -> npt.NDArray[np.float64]:
        """m_t_given_t in observation space of shape (size, n)."""
        return self.filtered_obs_samples.mean(axis=1)

    @cached_property
    def Q_t_given_t_obs(self) -> npt.NDArray[np.float64]:
        """Q_t_given_t in observation space of shape (size, n)."""
        return cov_from_samples(self.filtered_obs_samples)

    # misc

    def log_likelihoods(
        self,
        params_chain: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return -self.parent.neg_log_likelihoods(params_chain, self.y, self.n_sample)

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
        if with_obs_cov:
            raise NotImplementedError("with_obs_cov is not implemented in particle filter.")
        if n_std is None:
            raise NotImplementedError("CI with α-level is not implemented")
        else:
            m_t_given_t_obs = self.m_t_given_t_obs
            Q_t_given_t_obs = self.Q_t_given_t_obs

            filtered_obs_std = sqrt_diag(Q_t_given_t_obs)
            Δ = filtered_obs_std * n_std
            return m_t_given_t_obs - Δ, m_t_given_t_obs, m_t_given_t_obs + Δ

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
        # (size, N, n)
        x_t_forecast = self.get_forecast_obs(T_forecast, with_obs_cov=with_obs_cov)

        if n_std is None:
            raise NotImplementedError("CI with α-level is not implemented")
        else:
            forecast_obs = x_t_forecast.mean(axis=1)
            forecast_obs_cov = cov_from_samples(x_t_forecast)
            forecast_obs_std = sqrt_diag(forecast_obs_cov)
            return (
                forecast_obs - n_std * forecast_obs_std,
                forecast_obs,
                forecast_obs + n_std * forecast_obs_std,
            )

    # misc

    @cached_property
    def log_likelihood(self) -> float:
        return -self.parent.neg_log_likelihood_from_w_t(self.w_t)


# TODO: refactor s.t. both this and GenericLinearGaussianStateSpaceModel
# are based on an abstract class
@dataclass
class GenericParticleFilterStateSpaceModel(GenericLinearGaussianStateSpaceModel):
    """Generic particle filter state space model.

    :param particle_filter_method: Method to use for particle filter, valid options are
        bootstrap, particle_filter.
    :param transform_params: For MLE, function from unconstrainted to constrainted space
        i.e. from optimzation space to parameter space.
    :param untransform_params: For MLE, function from constrainted to unconstrainted space
        i.e. from parameter space to optimization space.
    """

    particle_filter_method: str = "bootstrap"
    transform_params: TransformFunc | None = None
    untransform_params: TransformFunc | None = None

    def __post_init__(self) -> None:
        if self.particle_filter_method not in ["bootstrap", "optimal"]:
            raise ValueError(f"Unknown particle filter method {self.particle_filter_method}")
        if (self.transform_params is None) is not (self.untransform_params is None):
            raise ValueError("transform_params and untransform_params has to be either both None or both not None")

    @cached_property
    def sampling_X0_inplace(self) -> SamplingX0Inplace:
        """In case it is not implemented and particle_filter need it in an if-clause that this is not executed anyway."""

        @jit("(f8[:, ::1], f8[::1], f8[::1])", nopython=True, nogil=True)
        def sampling_X0_inplace(
            x_0: npt.NDArray[np.float64],
            y_0: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
        ) -> None:
            raise NotImplementedError

        return sampling_X0_inplace

    @cached_property
    def sampling_X0_tilde_inplace(self) -> SamplingX0TildeInplace:
        """In case it is not implemented and particle_filter need it in an if-clause that this is not executed anyway."""

        @jit("(f8[:, ::1], f8[::1])", nopython=True, nogil=True)
        def sampling_X0_tilde_inplace(
            x_0_tilde: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
        ) -> None:
            raise NotImplementedError

        return sampling_X0_tilde_inplace

    @property
    def sampling_one_step_ahead_inplace(self) -> SamplingOneStepAheadInplace:
        raise NotImplementedError

    @cached_property
    def sampling_q_inplace_bootstrap(self) -> SamplingQInplace:
        sampling_one_step_ahead_inplace = self.sampling_one_step_ahead_inplace

        @jit("(f8[:, ::1], f8[:, ::1], f8[::1], f8[::1], i8)", nopython=True, nogil=True)
        def sampling_q_inplace_bootstrap(
            x_t_tilde: npt.NDArray[np.float64],
            x_tMinus1: npt.NDArray[np.float64],
            y_t: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            t: int,
        ) -> None:
            """Sampling from q distribution.

            :param x_t_tilde: output buffer.

            For SIR in general,
            or in the case of
            bootstrap particle filter,
            y_t is ignored.
            """
            sampling_one_step_ahead_inplace(x_t_tilde, x_tMinus1, params, t)

        return sampling_q_inplace_bootstrap

    @property
    def sampling_q_inplace_optimal(self) -> SamplingQInplace:
        raise NotImplementedError

    @property
    def sampling_q_inplace(self) -> SamplingQInplace:
        """Sampling from q distribution.

        :param x_t_tilde: output buffer.

        For SIR in general,
        or in the case of
        bootstrap particle filter,
        y_t is ignored.
        """
        return {
            "bootstrap": self.sampling_q_inplace_bootstrap,
        }[self.particle_filter_method]

    @property
    def weights_inplace_bootstrap(self) -> WeightsInplace:
        raise NotImplementedError

    @property
    def weights_inplace_optimal(self) -> WeightsInplace:
        raise NotImplementedError

    @property
    def weights_inplace(self) -> WeightsInplace:
        raise NotImplementedError

    @property
    def neg2_log_likelihoods(self) -> Any:
        raise NotImplementedError

    @cached_property
    def neg_log_likelihood(self) -> Likelihood:
        d = self.d
        sampling_X0_tilde_inplace = self.sampling_X0_tilde_inplace
        sampling_X0_inplace = self.sampling_X0_inplace
        sampling_q_inplace = self.sampling_q_inplace
        weights_inplace = self.weights_inplace
        use_bootstrap = self.particle_filter_method == "bootstrap"
        n_processes = get_num_threads()
        logger.info("Using %d processes for the calculation of log-likelihood", n_processes)

        @jit("(f8[::1], f8[::1], f8[:, ::1], i8)", nopython=True, nogil=True)
        def sum_w_inplace(
            s_t: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            N: int,
        ) -> None:
            """Negative log likelihood.

            Modified from particle_filter but simplified
            for likelihood calculation only.

            This does not speed up comparing to particle_filter,
            but reduce memory use roughly by a factor of
            size + 1 / (2 * d + 1).
            """
            size = y.shape[0]
            # initialize, keep the latest only
            temp = np.empty((2, N, d))
            # For bootstrap, x_t_tilde is x_t_given_tMinus1
            x_t_tilde = temp[0]
            # x_t is x_t_given_t
            x_t = temp[1]
            w_t = np.empty(N)

            t = 0
            if use_bootstrap:
                # initial generation
                sampling_X0_tilde_inplace(x_t_tilde, params)
                # weights
                # x_t is not used by weights_inplace_bootstrap
                weights_inplace(w_t, x_t_tilde, x_t, y[t], params, t)
                s = w_t.sum()
                # special case: when it is very diffused
                if s == 0.0:
                    w_t[:] = 1.0 / N
                    s_t[t] = 1.0
                else:
                    s_t[t] = s
                # resampling
                random_choice_weighted_inplace(x_t, x_t_tilde, N, w_t)
            else:
                # initial generation
                # x_t_tilde not generated in the guided particle filter case
                # guided particle filter does not have initial weights
                # as X0 is not resampled from some weights
                s_t[t] = 1.0
                sampling_X0_inplace(x_t, y[t], params)
            for t in range(1, size):
                # generation
                sampling_q_inplace(x_t_tilde, x_t, y[t], params, t)
                # weights
                weights_inplace(w_t, x_t_tilde, x_t, y[t], params, t)
                s = w_t.sum()
                # special case: when it is very diffused
                if s == 0.0:
                    w_t[:] = 1.0 / N
                    s_t[t] = 1.0
                else:
                    s_t[t] = s
                # resampling
                random_choice_weighted_inplace(x_t, x_t_tilde, N, w_t)

        @jit("f8(f8[::1], f8[:, ::1], i8)", nopython=True, nogil=True, cache=False, parallel=True)
        def neg_log_likelihood(
            params: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            N: int,
        ) -> float:
            size = y.shape[0]

            # memalloc
            s_i_t = np.empty((n_processes, size))

            # ceil
            n_sample_per_process = (N + n_processes - 1) // n_processes
            # this would be slightly larger than N if N is not divisible by n_processes
            N = n_sample_per_process * n_processes

            for i in prange(n_processes):
                sum_w_inplace(s_i_t[i], params, y, n_sample_per_process)
            return size * np.log(N) - np.log(s_i_t.sum(axis=0)).sum()

        return neg_log_likelihood

    @cached_property
    def neg_log_likelihoods(self) -> Likelihoods:
        d = self.d
        sampling_X0_tilde_inplace = self.sampling_X0_tilde_inplace
        sampling_X0_inplace = self.sampling_X0_inplace
        sampling_q_inplace = self.sampling_q_inplace
        weights_inplace = self.weights_inplace
        use_bootstrap = self.particle_filter_method == "bootstrap"

        @jit("f8(f8[::1], f8[:, ::1], i8)", nopython=True, nogil=True)
        def neg_log_likelihood(
            params: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            N: int,
        ) -> float:
            """Negative log likelihood.

            Modified from particle_filter but simplified
            for likelihood calculation only.

            This does not speed up comparing to particle_filter,
            but reduce memory use roughly by a factor of
            size + 1 / (2 * d + 1).
            """
            size = y.shape[0]
            # initialize, keep the latest only
            temp = np.empty((2, N, d))
            # For bootstrap, x_t_tilde is x_t_given_tMinus1
            x_t_tilde = temp[0]
            # x_t is x_t_given_t
            x_t = temp[1]
            w_t = np.empty(N)

            res = size * np.log(N)
            t = 0
            if use_bootstrap:
                # initial generation
                sampling_X0_tilde_inplace(x_t_tilde, params)
                # weights
                # x_t is not used by weights_inplace_bootstrap
                weights_inplace(w_t, x_t_tilde, x_t, y[t], params, t)
                s = w_t.sum()
                # special case: when it is very diffused
                if s == 0.0:
                    w_t[:] = 1.0 / N
                else:
                    res -= np.log(s)
                # resampling
                random_choice_weighted_inplace(x_t, x_t_tilde, N, w_t)
            else:
                # initial generation
                # x_t_tilde not generated in the guided particle filter case
                # guided particle filter does not have initial weights
                # as X0 is not resampled from some weights
                sampling_X0_inplace(x_t, y[t], params)
            for t in range(1, size):
                # generation
                sampling_q_inplace(x_t_tilde, x_t, y[t], params, t)
                # weights
                weights_inplace(w_t, x_t_tilde, x_t, y[t], params, t)
                s = w_t.sum()
                # special case: when it is very diffused
                if s == 0.0:
                    w_t[:] = 1.0 / N
                else:
                    res -= np.log(s)
                # resampling
                random_choice_weighted_inplace(x_t, x_t_tilde, N, w_t)

            return res

        @jit("f8[::1](f8[:, ::1], f8[:, ::1], i8)", nopython=True, nogil=True, cache=False, parallel=True)
        def neg_log_likelihoods(
            params: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            N: int,
        ) -> npt.NDArray[np.float64]:
            n_params = params.shape[0]

            # memalloc
            ls = np.empty(n_params)
            for i in prange(n_params):
                ls[i] = neg_log_likelihood(params[i], y, N)
            return ls

        return neg_log_likelihoods

    # TODO: refactor classes

    @property
    def mle_scipy_minimize(self) -> Mle:  # type: ignore[override]  # inheritance issue
        neg_log_likelihood = self.neg_log_likelihood
        transform_params = self.transform_params
        untransform_params = self.untransform_params

        if transform_params is not None:
            logger.info("Calculating MLE using transformed parameter space, possibly for unconstrainted optimization.")

            @jit("f8(f8[::1], f8[:, ::1], i8)", nopython=True, nogil=True, cache=False)
            def minimization_func(
                unconstrained: npt.NDArray[np.float64],
                y: npt.NDArray[np.float64],
                N: int,
            ) -> float:
                """Minimization function for scipy.optimize.minimize.

                :param unconstrained: unconstrained parameters in optimization space.
                """
                constrained = transform_params(unconstrained)  # type: ignore[misc]  # type limitation
                return neg_log_likelihood(constrained, y, N)

        else:
            minimization_func = neg_log_likelihood

        def mle_scipy_minimize(
            y: npt.NDArray[np.float64],
            params0: npt.NDArray[np.float64],
            n_iter: int,
            n_sample: int,
            return_chain: bool,
            **kwargs: dict[str, Any],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            """Calculate MLE from log-likelihood using scipy.optimize.minimize.

            Tips: include `bounds` for constrainted variables.
            """
            from scipy.optimize import minimize

            def _callback(unconstrained: npt.NDArray[np.float64]) -> None:
                """Store iterations."""
                chain.append(unconstrained if transform_params is None else transform_params(unconstrained))

            if n_iter:
                logger.warning("n_iter %s is ignored when using scipy.optimize.minimize.", n_iter)

            chain: list[npt.NDArray[np.float64]] = []
            callack: Callable[[npt.NDArray[np.float64]], None] | None
            if return_chain:
                chain.append(params0)
                callback = _callback
            else:
                callback = None

            opt = minimize(
                minimization_func,
                params0 if untransform_params is None else untransform_params(params0),
                args=(y, n_sample),
                callback=callback,
                **kwargs,
            )
            msg = opt.message
            if not opt.success:
                raise RuntimeError(f"MLE unccessful with message: {msg}")
            logger.info("MLE constructed with message: %s", msg)
            return opt.x if transform_params is None else transform_params(opt.x), np.array(chain)

        return mle_scipy_minimize

    @staticmethod
    def params_guess(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Estimate initial parameters using the data with law of large number."""
        raise NotImplementedError

    def fit(  # type: ignore[override]  # inheritance issue
        self,
        y: npt.ArrayLike,
        params: npt.ArrayLike | None,
        mle_method: str = "",
        mle_n_iter: int = 0,
        mle_return_chain: bool = False,
        mle_kwargs: dict[Any, Any] = {},
        n_sample: int = 0,
        t: npt.NDArray[Any] | None = None,
        param_names: list[str] | None = None,
    ) -> GenericParticleFilterStateSpaceModelFit:
        """Return a GenericParticleFilterStateSpaceModelFit instance.

            :param y: observed data (Y)
            :param params: parameters of the model
            :param mle_method: method used to estimate MLE
            :param mle_n_iter: number of iteration run with MLE
            :param mle_return_chain: if True, return the chain of parameters during MLE
            :param t: optional time axis
            :param param_names: optional names of parameters

        TODO: tailor for particle filter
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        if params is None:
            logger.info("Estimating initial parameters using the data with law of large number...")
            params = self.params_guess(y)
        # using default if exists
        if not mle_kwargs and hasattr(self, "mle_kwargs"):
            mle_kwargs = self.mle_kwargs  # type: ignore[attr-defined]
        return GenericParticleFilterStateSpaceModelFit(
            self,
            y,
            params,
            mle_method=mle_method,
            mle_n_iter=mle_n_iter,
            mle_return_chain=mle_return_chain,
            mle_kwargs=mle_kwargs,
            n_sample=n_sample,
            t=t,
            param_names=param_names or self.param_names,
        )

    @property
    def smoothed_state_to_obs(self) -> SmoothedStateToObs:  # type: ignore[override]  # inheritance issue
        raise NotImplementedError

    @cached_property
    def forecast(self) -> Forecast:  # type: ignore[override]  # inheritance issue
        sampling_one_step_ahead_inplace = self.sampling_one_step_ahead_inplace

        @jit(
            "f8[:, :, ::1](f8[::1], f8[:, :, ::1], int64)",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def forecast(
            params: npt.NDArray[np.float64],
            x_t: npt.NDArray[np.float64],
            T_forecast: int,
        ) -> npt.NDArray[np.float64]:
            size, N, d = x_t.shape

            # memalloc
            x_s_forecast = np.empty((T_forecast, N, d))

            # initialize
            x_prev = x_t[-1]
            # s from 0 to T_forecast - 1
            for s in range(T_forecast):
                # t from T + 1 to T + T_forecast
                t = s + size
                sampling_one_step_ahead_inplace(x_s_forecast[s], x_prev, params, t)
                x_prev = x_s_forecast[s]
            return x_s_forecast

        return forecast

    @cached_property
    def neg_log_likelihood_from_w_t(self) -> LikelihoodFromWeights:
        @jit("f8(f8[:, ::1])", nopython=True, nogil=True)
        def neg_log_likelihood_from_w_t(w_t: npt.NDArray[np.float64]) -> float:
            size, N = w_t.shape
            return size * np.log(N) - np.log(w_t.sum(axis=1)).sum()

        return neg_log_likelihood_from_w_t

    @cached_property
    def particle_filter(self) -> ParticleFilter:
        d = self.d
        sampling_X0_tilde_inplace = self.sampling_X0_tilde_inplace
        sampling_X0_inplace = self.sampling_X0_inplace
        sampling_q_inplace = self.sampling_q_inplace
        weights_inplace = self.weights_inplace
        use_bootstrap = self.particle_filter_method == "bootstrap"

        @jit("Tuple((f8[:, :, ::1], f8[:, :, ::1], f8[:, ::1]))(f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True)
        def particle_filter(
            y: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            N: int,
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            """Guided particle filter.

            :param N: number of particles.
            """
            size = y.shape[0]
            # initialize
            temp = np.empty((2, size, N, d))
            # For bootstrap, x_t_tilde is x_t_given_tMinus1
            x_t_tilde = temp[0]
            # x_t is x_t_given_t
            x_t = temp[1]
            w_t = np.empty((size, N))

            t = 0
            if use_bootstrap:
                # initial generation
                sampling_X0_tilde_inplace(x_t_tilde[t], params)
                # weights
                # x_t[t-1] is not used by weights_inplace_bootstrap
                weights_inplace(w_t[t], x_t_tilde[t], x_t[t - 1], y[t], params, t)
                # special case: when it is very diffused
                if np.count_nonzero(w_t[t]) == 0:
                    w_t[t] = 1.0 / N
                # resampling
                random_choice_weighted_inplace(x_t[t], x_t_tilde[t], N, w_t[t])
            else:
                # initial generation
                # x_t_tilde not generated in the guided particle filter case
                x_t_tilde[t] = np.nan
                # guided particle filter does not have initial weights
                # as X0 is not resampled from some weights
                w_t[t] = 1.0 / N
                sampling_X0_inplace(x_t[t], y[t], params)
            for t in range(1, size):
                # generation
                sampling_q_inplace(x_t_tilde[t], x_t[t - 1], y[t], params, t)
                # weights
                weights_inplace(w_t[t], x_t_tilde[t], x_t[t - 1], y[t], params, t)
                # special case: when it is very diffused
                if np.count_nonzero(w_t[t]) == 0:
                    w_t[t] = 1.0 / N
                # resampling
                random_choice_weighted_inplace(x_t[t], x_t_tilde[t], N, w_t[t])
            return x_t_tilde, x_t, w_t

        return particle_filter


@dataclass
class GeneralLinearGaussianParticleFilterStateSpaceModel(  # type: ignore[misc]  # inheritance issue in method forecast
    GenericParticleFilterStateSpaceModel, GeneralLinearGaussianStateSpaceModel
):
    """General particle filter in the special case of linear Gaussian state space model."""

    def __post_init__(self) -> None:
        GeneralLinearGaussianStateSpaceModel.__post_init__(self)
        GenericParticleFilterStateSpaceModel.__post_init__(self)
        # in this model X0 is the only way to pass μ_0, Γ_0
        assert self.X0 is not None
        self.X0: Callable[[npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]

    @cached_property
    def sampling_X0_inplace(self) -> SamplingX0Inplace:
        B0 = self.B0
        R0 = self.R0
        X0 = self.X0
        d = self.d

        @jit("(f8[:, ::1], f8[::1], f8[::1])", nopython=True, nogil=True)
        def sampling_X0_inplace(
            x_0: npt.NDArray[np.float64],
            y_0: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
        ) -> None:
            """Sampling X_0.

            :param x_0: output buffer of shape (d, N).

            For SIR in general,
            or bootstrap particle filter
            when prior of X_0 contains no information (defused.)
            """
            N = x_0.shape[0]
            # initialize
            μ, Γ = X0(params)
            B = B0(params)
            R = R0(params)

            BΓB_R = B @ Γ @ B.T + R

            m = μ + Γ @ B.T @ np.linalg.solve(BΓB_R, y_0 - B @ μ)
            Q = Γ - Γ @ B.T @ np.linalg.solve(BΓB_R, B @ Γ)
            if d == 1:
                x_0[:] = np.random.normal(m[0], np.sqrt(Q[0, 0]), N).reshape(N, 1)
            else:
                with objmode():
                    x_0[:] = np.random.multivariate_normal(m, Q, N)

        return sampling_X0_inplace

    @cached_property
    def sampling_X0_tilde_inplace(self) -> SamplingX0TildeInplace:
        X0 = self.X0
        d = self.d

        @jit("(f8[:, ::1], f8[::1])", nopython=True, nogil=True)
        def sampling_X0_tilde_inplace(
            x_0_tilde: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
        ) -> None:
            """Sampling X_0.

            :param x_0_tilde: output buffer.

            This is only used for bootstrap particle filter.
            """
            N = x_0_tilde.shape[0]
            # initialize
            μ, Γ = X0(params)
            if d == 1:
                x_0_tilde[:] = np.random.normal(μ[0], np.sqrt(Γ[0, 0]), N).reshape(N, 1)
            else:
                with objmode():
                    x_0_tilde[:] = np.random.multivariate_normal(μ, Γ, N)

        return sampling_X0_tilde_inplace

    @cached_property
    def sampling_one_step_ahead_inplace(self) -> SamplingOneStepAheadInplace:
        A0 = self.A0
        Σ0 = self.Σ0
        A = self.A
        Σ = self.Σ
        d = self.d
        # just for passing to multivariate_normal
        μ_zeros = np.zeros(d)

        @jit("(f8[:, ::1], f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True)
        def sampling_one_step_ahead_inplace(
            x_t_given_tMinus1: npt.NDArray[np.float64],
            x_tMinus1: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            t: int,
        ) -> None:
            """One-step ahead prediction.

            :param x_t_given_tMinus1: output buffer.

            For forcasting in the case of SIR.
            For Bootstrap particle filter,
            same as sampling_q_inplace
            where x_t_given_tMinus1 is x_t_tilde.

            x_tMinus1 is x_tMinus1_given_tMinus1.
            """
            N = x_t_given_tMinus1.shape[0]
            # initialize
            A_t = A0(params)
            if A is not None:
                A(A_t, params, t)
            Σ_t = Σ0(params)
            if Σ is not None:
                Σ(Σ_t, params, t)

            for i in range(N):
                # mean
                x_t_given_tMinus1[i] = A_t @ x_tMinus1[i]
            if d == 1:
                x_t_given_tMinus1 += np.random.normal(0.0, np.sqrt(Σ_t[0, 0]), N).reshape(N, 1)
            else:
                with objmode():
                    x_t_given_tMinus1 += np.random.multivariate_normal(μ_zeros, Σ_t, N)

        return sampling_one_step_ahead_inplace

    @cached_property
    def sampling_q_inplace_optimal(self) -> SamplingQInplace:
        A0 = self.A0
        B0 = self.B0
        Σ0 = self.Σ0
        R0 = self.R0
        A = self.A
        B = self.B
        Σ = self.Σ
        R = self.R
        d = self.d
        # just for passing to multivariate_normal
        μ_zeros = np.zeros(d)

        @jit("(f8[:, ::1], f8[:, ::1], f8[::1], f8[::1], i8)", nopython=True, nogil=True)
        def sampling_q_inplace_optimal(
            x_t_tilde: npt.NDArray[np.float64],
            x_tMinus1: npt.NDArray[np.float64],
            y_t: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            t: int,
        ) -> None:
            """Sampling from q distribution.

            :param x_t_tilde: output buffer.

            For SIR in general,
            or in the case of
            bootstrap particle filter,
            y_t is ignored.
            """
            N = x_t_tilde.shape[0]
            # initialize
            A_t = A0(params)
            if A is not None:
                A(A_t, params, t)
            B_t = B0(params)
            if B is not None:
                B(B_t, params, t)
            Σ_t = Σ0(params)
            if Σ is not None:
                Σ(Σ_t, params, t)
            R_t = R0(params)
            if R is not None:
                R(R_t, params, t)

            BA = B_t @ A_t
            BΣB_R = B_t @ Σ_t @ B_t.T + R_t
            ΣB = Σ_t @ B_t.T
            BΣ = ΣB.T

            for i in range(N):
                x_t_tilde[i] = A_t @ x_tMinus1[i] + ΣB @ np.linalg.solve(BΣB_R, y_t - BA @ x_tMinus1[i])
            Q = Σ_t - ΣB @ np.linalg.solve(BΣB_R, BΣ)
            if d == 1:
                x_t_tilde += np.random.normal(0.0, np.sqrt(Q[0, 0]), N).reshape(N, 1)
            else:
                with objmode():
                    x_t_tilde += np.random.multivariate_normal(μ_zeros, Q, N)

        return sampling_q_inplace_optimal

    @property
    def sampling_q_inplace(self) -> SamplingQInplace:
        """Sampling from q distribution.

        :param x_t_tilde: output buffer.

        For SIR in general,
        or in the case of
        bootstrap particle filter,
        y_t is ignored.
        """
        return {
            "bootstrap": self.sampling_q_inplace_bootstrap,
            "optimal": self.sampling_q_inplace_optimal,
        }[self.particle_filter_method]

    @cached_property
    def weights_inplace_bootstrap(self) -> WeightsInplace:
        B0 = self.B0
        R0 = self.R0
        B = self.B
        R = self.R

        @jit("(f8[::1], f8[:, ::1], f8[:, ::1], f8[::1], f8[::1], i8)", nopython=True, nogil=True)
        def weights_inplace_bootstrap(
            w_t: npt.NDArray[np.float64],
            x_t_tilde: npt.NDArray[np.float64],
            x_tMinus1: npt.NDArray[np.float64],
            y_t: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            t: int,
        ) -> None:
            """Calculate unnormalized weights.

            :param w_t: output buffer.

            For SIR in general.
            In the case of
            bootstrap particle filter,
            x_tMinus1 is ignored.
            In the case of
            optimal guided particle filter,
            x_t_tilde is ignored.
            """
            N = w_t.shape[0]
            # initialize
            B_t = B0(params)
            if B is not None:
                B(B_t, params, t)
            R_t = R0(params)
            if R is not None:
                R(R_t, params, t)
            # somehow by default the pinv return Fortran order
            R_t_inv = np.linalg.pinv(R_t).T
            for i in range(N):
                w_t[i] = prob_normal_each(
                    y_t,
                    B_t @ x_t_tilde[i],
                    R_t_inv,
                    False,
                )

        return weights_inplace_bootstrap

    @cached_property
    def weights_inplace_optimal(self) -> WeightsInplace:
        A0 = self.A0
        B0 = self.B0
        Σ0 = self.Σ0
        R0 = self.R0
        A = self.A
        B = self.B
        Σ = self.Σ
        R = self.R

        @jit("(f8[::1], f8[:, ::1], f8[:, ::1], f8[::1], f8[::1], i8)", nopython=True, nogil=True)
        def weights_inplace_optimal(
            w_t: npt.NDArray[np.float64],
            x_t_tilde: npt.NDArray[np.float64],
            x_tMinus1: npt.NDArray[np.float64],
            y_t: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            t: int,
        ) -> None:
            """Calculate unnormalized weights.

            :param w_t: output buffer.

            For SIR in general.
            In the case of
            bootstrap particle filter,
            x_tMinus1 is ignored.
            In the case of
            optimal guided particle filter,
            x_t_tilde is ignored.
            """
            N = w_t.shape[0]
            # initialize
            A_t = A0(params)
            if A is not None:
                A(A_t, params, t)
            B_t = B0(params)
            if B is not None:
                B(B_t, params, t)
            Σ_t = Σ0(params)
            if Σ is not None:
                Σ(Σ_t, params, t)
            R_t = R0(params)
            if R is not None:
                R(R_t, params, t)

            BA = B_t @ A_t
            # somehow by default the pinv return Fortran order
            BΣB_R_inv = np.linalg.pinv(B_t @ Σ_t @ B_t.T + R_t).T
            for i in range(N):
                w_t[i] = prob_normal_each(
                    y_t,
                    BA @ x_tMinus1[i],
                    BΣB_R_inv,
                    False,
                )

        return weights_inplace_optimal

    @property
    def weights_inplace(self) -> WeightsInplace:
        """Calculate unnormalized weights.

        :param w_t: output buffer.

        For SIR in general.
        In the case of
        bootstrap particle filter,
        x_tMinus1 is ignored.
        In the case of
        optimal guided particle filter,
        x_t_tilde is ignored.
        """
        return {
            "bootstrap": self.weights_inplace_bootstrap,
            "optimal": self.weights_inplace_optimal,
        }[self.particle_filter_method]

    # TODO: refactor classes
    # We really only need __post_init__ from GeneralLinearGaussianStateSpaceModel

    @cached_property
    def kalman_filtering_inplace(self) -> Any:
        raise NotImplementedError

    @cached_property
    def kalman_filtering(self) -> Any:
        raise NotImplementedError

    @cached_property
    def kalman_smoothing_inplace(self) -> Any:
        raise NotImplementedError

    @cached_property
    def kalman_smoothing(self) -> Any:
        raise NotImplementedError

    @cached_property
    def backward_sampling(self) -> Any:
        raise NotImplementedError

    @cached_property
    def gibbs_sampling(self) -> Any:
        raise NotImplementedError

    @cached_property
    def smoothed_state_to_obs(self) -> SmoothedStateToObs:  # type: ignore[override]  # inheritance issue
        B0 = self.B0
        B = self.B
        B0_forecast = self.B0_forecast
        B_forecast = self.B_forecast
        n = self.n

        @jit(
            "f8[:, :, ::1](f8[::1], f8[:, :, ::1], int64)",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def smoothed_state_to_obs(
            params: npt.NDArray[np.float64],
            x_t: npt.NDArray[np.float64],
            t0: int,
        ) -> npt.NDArray[np.float64]:
            size, N, _ = x_t.shape

            # memalloc
            B_t = B0(params)
            B_t_forecast = B_t if B0_forecast is None else B0_forecast(params)
            smoothed_obs = np.empty((size, N, n))

            for t in range(size):
                if B is not None:
                    B(B_t, params, t0 + t)
                if B0_forecast is not None and B_forecast is not None:
                    B_forecast(B_t_forecast, params, t0 + t)
                for i in range(N):
                    smoothed_obs[t, i] = B_t_forecast @ x_t[t, i]
            return smoothed_obs

        return smoothed_state_to_obs

    @cached_property
    def neg2_log_likelihood(self) -> Any:
        raise NotImplementedError


@dataclass
class GeneralNonLinearBootstrapParticleFitlerStateSpaceModel(GenericParticleFilterStateSpaceModel):
    r"""General non-linear bootstrap particle filter state space model.

    :param pdf_Y_inplace: `pdf_Y_inplace(pdf, y, x, params, t)` writes the probability density of `y` given `x` at time `t` to `pdf`.
    :param K_t_inplace: `K_t_inplace(x_t, x_tMinus1, u_t, params, t)` computes and write to `x_t` in terms of `x_tMinus1` and `u_t`, where `u_t` is generated below.
    :param sampling_X0_tilde_inplace: `sampling_X0_tilde_inplace(x_0_tilde, params)` writes samples from `X_0` to `x_0_tilde`.
    :param sampling_U_t: `sampling_U_t(params, t, N)` returns samples from `U_t` to `u_t` as `u_t`.
    : param smoothed_state_to_obs: `state_to_obs(params, x_t, t0)` returns the state in observation space.

    This implements general non-linear bootstrap particle filter state space model
    according to the specification, w.r.t. the four callable above respectively.

    .. math::

        Y_t | X_t = x_t \sim f_{V_t}(\cdotp; x_t)

        X_t = K_t(X_{t-1}, U_t)

        X_0 \sim f_{X_0}

        U_t \sim f_{U_t}
    """
    mle_kwargs: dict[Any, Any] = field(default_factory=dict)
    param_names: list[str] | None = None
    d: int = 1
    n: int = 1
    pdf_Y_inplace: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], int], None
    ] = None  # type:ignore[assignment]  # we don't really allow None here
    K_t_inplace: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], int], None
    ] = None  # type:ignore[assignment]  # we don't really allow None here
    sampling_X0_tilde_inplace: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64]], None
    ] = None  # type:ignore[assignment]  # we don't really allow None here
    sampling_U_t: Callable[
        [npt.NDArray[np.float64], int, int], npt.NDArray[np.float64]
    ] = None  # type:ignore[assignment]  # we don't really allow None here
    state_to_obs: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64], int], npt.NDArray[np.float64]
    ] = None  # type:ignore[assignment]  # we don't really allow None here

    @cached_property
    def sampling_one_step_ahead_inplace(self) -> SamplingOneStepAheadInplace:
        K_t_inplace = self.K_t_inplace
        sampling_U_t = self.sampling_U_t

        @jit("(f8[:, ::1], f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True)
        def sampling_one_step_ahead_inplace(
            x_t_given_tMinus1: npt.NDArray[np.float64],
            x_tMinus1: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            t: int,
        ) -> None:
            """One-step ahead prediction.

            :param x_t_given_tMinus1: output buffer.
            """
            N = x_t_given_tMinus1.shape[0]
            u_t = sampling_U_t(params, t, N)
            K_t_inplace(x_t_given_tMinus1, x_tMinus1, u_t, params, t)

        return sampling_one_step_ahead_inplace

    @cached_property
    def weights_inplace_bootstrap(self) -> WeightsInplace:
        pdf_Y_inplace = self.pdf_Y_inplace

        @jit("(f8[::1], f8[:, ::1], f8[:, ::1], f8[::1], f8[::1], i8)", nopython=True, nogil=True)
        def weights_inplace_bootstrap(
            w_t: npt.NDArray[np.float64],
            x_t_tilde: npt.NDArray[np.float64],
            x_tMinus1: npt.NDArray[np.float64],
            y_t: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            t: int,
        ) -> None:
            """Calculate unnormalized weights.

            :param w_t: output buffer.

            x_tMinus1 is ignored.
            """
            pdf_Y_inplace(w_t, y_t, x_t_tilde, params, t)

        return weights_inplace_bootstrap

    @property
    def weights_inplace(self) -> WeightsInplace:
        """Calculate unnormalized weights.

        :param w_t: output buffer.

        For SIR in general.
        In the case of
        bootstrap particle filter,
        x_tMinus1 is ignored.
        In the case of
        optimal guided particle filter,
        x_t_tilde is ignored.
        """
        return {
            "bootstrap": self.weights_inplace_bootstrap,
        }[self.particle_filter_method]

    @property
    def smoothed_state_to_obs(self) -> SmoothedStateToObs:  # type: ignore[override]  # inheritance issue
        return self.state_to_obs
