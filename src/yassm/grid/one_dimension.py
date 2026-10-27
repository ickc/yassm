"""General 1-dimensional state space model."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
from numba import jit

from ..lib.grid.one_dimension import (
    filter_update_inplace,
    get_grid_to_position,
    one_step_ahead_inplace,
    ps_to_mQ,
    smoothing_backward_recursion_inplace,
)
from ..util import as_c_array, as_scipy_objective, deprecated_aliases

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import numpy.typing as npt
    from typing_extensions import Protocol

    from ..lib.grid.one_dimension import DesignFunc, TransitionFunc

    KalmanFilteringInplace = Callable[
        [
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
        ],
        tuple[
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
        ],
        None,
    ]
    KalmanSmoothing = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
        npt.NDArray[np.float64],
    ]
    Likelihood = Callable[
        [
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
        ],
        npt.NDArray[np.float64],
    ]

    class Mle(Protocol):
        def __call__(
            self,
            y: npt.NDArray[np.float64],
            params0: npt.NDArray[np.float64],
            p_0: npt.NDArray[np.float64],
            n_iter: int,
            return_chain: bool,
            **kwargs: dict[str, Any],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


logger = getLogger(__name__)


@dataclass
class GridFilterFit:
    parent: GridFilterModel
    y: npt.NDArray[np.float64]
    params: npt.NDArray[np.float64]
    p_0: npt.NDArray[np.float64]
    mle_method: str = "scipy_minimize"
    mle_n_iter: int = 0
    mle_return_chain: bool = False
    mle_kwargs: dict[Any, Any] = field(default_factory=dict)
    t: npt.NDArray[Any] | None = None
    param_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.y = as_c_array(self.y)
        self.params = as_c_array(self.params)
        self.p_0 = as_c_array(self.p_0)
        self.size = self.y.size
        self.G = self.p_0.size
        self.N = self.params.size

        if self.mle_n_iter:
            self.params, self.mle_params_chain = self.parent.mle(self.mle_method)(
                self.y,
                self.params,
                self.p_0,
                self.mle_n_iter,
                self.mle_return_chain,
                **self.mle_kwargs,
            )

    # defaults

    @property
    def t_array(self) -> npt.ArrayLike:
        return range(self.size) if self.t is None else self.t

    @cached_property
    def x_array(self) -> npt.NDArray[np.float64]:
        return np.linspace(self.parent.x_0, self.parent.x_f, self.G)

    # running functions and cache: _run_*

    def _run_kalman_filtering(self) -> None:
        logger.info("Running kalman filtering...")
        (
            self._p_t_given_tMinus1,
            self._p_t_given_t,
        ) = self.parent.kalman_filtering(
            self.y,
            self.params,
            self.p_0,
        )

    def _run_kalman_smoothing(self) -> None:
        logger.info("Running kalman smoothing...")
        self._p_t_given_T = self.parent.kalman_smoothing(
            self.params,
            self.p_t_given_tMinus1,
            self.p_t_given_t,
        )

    # cached parameters from _run_*

    @property
    def p_t_given_tMinus1(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_p_t_given_tMinus1"):
            self._run_kalman_filtering()
        return self._p_t_given_tMinus1

    @property
    def p_t_given_t(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_p_t_given_t"):
            self._run_kalman_filtering()
        return self._p_t_given_t

    @property
    def p_t_given_T(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_p_t_given_T"):
            self._run_kalman_smoothing()
        return self._p_t_given_T

    def _run_p_t_given_t_to_mQ(self) -> None:
        mQs = ps_to_mQ(self.p_t_given_t, self.x_array)
        self._m_t_given_t = np.ascontiguousarray(mQs[:, 0])
        self._Q_t_given_t = np.ascontiguousarray(mQs[:, 1])

    @property
    def m_t_given_t(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_m_t_given_t"):
            self._run_p_t_given_t_to_mQ()
        return self._m_t_given_t

    @property
    def Q_t_given_t(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_Q_t_given_t"):
            self._run_p_t_given_t_to_mQ()
        return self._Q_t_given_t

    def _run_p_t_given_T_to_mQ(self) -> None:
        mQs = ps_to_mQ(self.p_t_given_T, self.x_array)
        self._m_t_given_T = np.ascontiguousarray(mQs[:, 0])
        self._Q_t_given_T = np.ascontiguousarray(mQs[:, 1])

    @property
    def m_t_given_T(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_m_t_given_T"):
            self._run_p_t_given_T_to_mQ()
        return self._m_t_given_T

    @property
    def Q_t_given_T(self) -> npt.NDArray[np.float64]:
        if not hasattr(self, "_Q_t_given_T"):
            self._run_p_t_given_T_to_mQ()
        return self._Q_t_given_T

    # other cached properties

    @cached_property
    def neg_log_likelihood(self) -> float:
        return self.parent.neg_log_likelihood(self.params, self.y, self.p_0)

    @property
    def log_likelihood(self) -> float:
        return -self.neg_log_likelihood


@dataclass
class GridFilterModel:
    N: int
    transition_func: TransitionFunc
    design_func: DesignFunc
    # grid spec
    x_0: float
    x_f: float
    G: int
    param_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.Δx = (self.x_f - self.x_0) / (self.G - 1)
        self.grid_to_position = get_grid_to_position(self.x_0, self.x_f, self.G)

    @cached_property
    def kalman_filtering_inplace(self) -> KalmanFilteringInplace:
        design_func = self.design_func
        grid_to_position = self.grid_to_position
        transition_func = self.transition_func
        Δx = self.Δx

        @jit(
            "(f8[:, ::1], f8[:, ::1], f8[::1], f8[::1], f8[::1])",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def kalman_filtering_inplace(
            p_t_given_tMinus1: npt.NDArray[np.float64],
            p_t_given_t: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            p_0: npt.NDArray[np.float64],
        ) -> None:
            """

            Parameters
            ----------
            p_t_given_tMinus1
                out array.
            p_t_given_t
                out array.
            y
                data.
            params
                params.
            p_0
                initial density.

            Returns
            -------
            None
                p_t_given_tMinus1, p_t_given_t
            """
            size = y.shape[0]

            # initialize
            p_t_given_tMinus1[0] = p_0

            t = 0
            y_t = y[t]
            filter_update_inplace(
                p_t_given_t[t],
                y_t,
                params,
                design_func,
                grid_to_position,
                p_t_given_tMinus1[t],
                Δx,
            )
            for t in range(1, size):
                tMinus1 = t - 1
                one_step_ahead_inplace(
                    p_t_given_tMinus1[t],
                    params,
                    transition_func,
                    grid_to_position,
                    p_t_given_t[tMinus1],
                    Δx,
                )
                y_t = y[t]
                filter_update_inplace(
                    p_t_given_t[t],
                    y_t,
                    params,
                    design_func,
                    grid_to_position,
                    p_t_given_tMinus1[t],
                    Δx,
                )

        return kalman_filtering_inplace

    @cached_property
    def kalman_filtering(self) -> KalmanFiltering:
        kalman_filtering_inplace = self.kalman_filtering_inplace

        @jit(
            "Tuple((f8[:, ::1], f8[:, ::1]))(f8[::1], f8[::1], f8[::1])",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def kalman_filtering(
            y: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            p_0: npt.NDArray[np.float64],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            size = y.shape[0]
            G = p_0.size

            # memalloc
            temp = np.zeros((2, size, G))
            p_t_given_tMinus1 = temp[0]
            p_t_given_t = temp[1]

            kalman_filtering_inplace(
                p_t_given_tMinus1,
                p_t_given_t,
                y,
                params,
                p_0,
            )
            return p_t_given_tMinus1, p_t_given_t

        return kalman_filtering

    @cached_property
    def kalman_smoothing_inplace(self) -> KalmanSmoothingInplace:
        transition_func = self.transition_func
        grid_to_position = self.grid_to_position
        Δx = self.Δx

        @jit(
            "(f8[:, ::1], f8[::1], f8[:, ::1], f8[:, ::1])",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def kalman_smoothing_inplace(
            p_t_given_T: npt.NDArray[np.float64],
            params: npt.NDArray[np.float64],
            p_t_given_tMinus1: npt.NDArray[np.float64],
            p_t_given_t: npt.NDArray[np.float64],
        ) -> None:
            size = p_t_given_T.shape[0]
            T = size - 1

            # initialize
            p_t_given_T[T] = p_t_given_t[T]

            # s = t, t = T
            for s in range(T - 1, -1, -1):
                sPlus1 = s + 1

                smoothing_backward_recursion_inplace(
                    p_t_given_T[s],
                    params,
                    transition_func,
                    grid_to_position,
                    p_t_given_t[s],
                    p_t_given_T[sPlus1],
                    p_t_given_tMinus1[sPlus1],
                    Δx,
                )

        return kalman_smoothing_inplace

    @cached_property
    def kalman_smoothing(self) -> KalmanSmoothing:
        kalman_smoothing_inplace = self.kalman_smoothing_inplace

        @jit(
            "f8[:, ::1](f8[::1], f8[:, ::1], f8[:, ::1])",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def kalman_smoothing(
            params: npt.NDArray[np.float64],
            p_t_given_tMinus1: npt.NDArray[np.float64],
            p_t_given_t: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            # memalloc
            p_t_given_T = np.zeros_like(p_t_given_t)

            kalman_smoothing_inplace(
                p_t_given_T,
                params,
                p_t_given_tMinus1,
                p_t_given_t,
            )
            return p_t_given_T

        return kalman_smoothing

    @cached_property
    def smoothed_state_to_obs(self) -> Any:
        raise NotImplementedError

    @cached_property
    def forecast(self) -> Any:
        raise NotImplementedError

    @cached_property
    def neg_log_likelihood(self) -> Likelihood:
        """Calculate negative log-likelihood.

        Y|θ
        """
        design_func = self.design_func
        grid_to_position = self.grid_to_position
        transition_func = self.transition_func
        Δx = self.Δx
        log_Δx = np.log(Δx)

        @jit(
            "f8(f8[::1], f8[::1], f8[::1])",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def neg_log_likelihood(
            params: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            p_0: npt.NDArray[np.float64],
        ) -> float:
            """

            Parameters
            ----------
            y
                data.
            params
                params.
            p_0
                initial density.

            Returns
            -------
            float
                negative log-likelihood up to a constant

            Notes
            -----
            Modified from Kalman filter.
            """
            size = y.shape[0]
            T = size - 1
            G = p_0.size

            # memalloc
            # keep only p at the current t step
            temp = np.empty((2, G))
            p_t_given_tMinus1 = temp[0]
            p_t_given_t = temp[1]

            # initialize
            p_t_given_tMinus1[:] = p_0

            # ignoring the initial defused distribution
            # likelihood in continuum limit is a density
            # this scaling factor ensure the result match
            # theoretical density e.g. in linear Gaussian case
            res = T * log_Δx

            t = 0
            y_t = y[t]
            p_t_given_t[:] = 0.0
            filter_update_inplace(
                p_t_given_t,
                y_t,
                params,
                design_func,
                grid_to_position,
                p_t_given_tMinus1,
                Δx,
            )
            for t in range(1, size):
                p_t_given_tMinus1[:] = 0.0
                one_step_ahead_inplace(
                    p_t_given_tMinus1,
                    params,
                    transition_func,
                    grid_to_position,
                    p_t_given_t,
                    Δx,
                )
                y_t = y[t]
                p_t_given_t[:] = 0.0
                filter_update_inplace(
                    p_t_given_t,
                    y_t,
                    params,
                    design_func,
                    grid_to_position,
                    p_t_given_tMinus1,
                    Δx,
                )
                integral = 0.0
                for g in range(G):
                    x = grid_to_position(g)
                    integral += design_func(y_t, x, params, Δx) * p_t_given_tMinus1[g]
                # negative log likelihood
                res -= np.log(integral)
            return res

        return neg_log_likelihood

    @cached_property
    def neg_log_likelihoods(self) -> Likelihoods:
        neg_log_likelihood = self.neg_log_likelihood

        @jit(
            "f8[::1](f8[:, ::1], f8[::1], f8[::1])",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def neg_log_likelihoods(
            paramses: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            p_0: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            N = paramses.shape[0]
            res = np.empty(N)
            for i in range(N):
                res[i] = neg_log_likelihood(
                    paramses[i],
                    y,
                    p_0,
                )
            return res

        return neg_log_likelihoods

    def mle(self, method: str = "scipy_minimize") -> Mle:
        """
        Similar to `yassm.linear_gaussian.LinearGaussianModel.mle`.
        """
        method_lower = method.lower()
        if method_lower != "scipy_minimize":
            logger.warning(
                "The only implemented MLE method is scipy_minimize for general linear Gaussian model. Using"
                " scipy_minimize..."
            )
        neg_log_likelihood = self.neg_log_likelihood

        def mle_scipy_minimize(
            y: npt.NDArray[np.float64],
            params0: npt.NDArray[np.float64],
            p_0: npt.NDArray[np.float64],
            n_iter: int,
            return_chain: bool,
            **kwargs: dict[str, Any],
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            """Calculate MLE from log-likelihood using scipy.optimize.minimize.

            Tips: include `bounds` for constrained variables.
            """
            from scipy.optimize import minimize

            def _callback(xk: npt.NDArray[np.float64]) -> None:
                """Store iterations."""
                chain.append(np.array(xk))

            if n_iter > 1:
                logger.warning("n_iter %s is ignored when using scipy.optimize.minimize.", n_iter)

            chain: list[npt.NDArray[np.float64]] = []
            callback: Callable[[npt.NDArray[np.float64]], None] | None
            if return_chain:
                chain.append(params0)
                callback = _callback
            else:
                callback = None

            opt = minimize(as_scipy_objective(neg_log_likelihood), params0, args=(y, p_0), callback=callback, **kwargs)
            msg = opt.message
            if not opt.success:
                raise RuntimeError(f"MLE unsuccessful with message: {msg}")
            logger.info("MLE constructed with message: %s", msg)
            return opt.x, np.array(chain)

        return mle_scipy_minimize

    def fit(
        self,
        y: npt.NDArray[np.float64],
        params: npt.NDArray[np.float64],
        p_0: npt.NDArray[np.float64],
        mle_method: str = "scipy_minimize",
        mle_n_iter: int = 0,
        mle_return_chain: bool = False,
        mle_kwargs: dict[Any, Any] | None = None,
        t: npt.NDArray[Any] | None = None,
        param_names: list[str] | None = None,
    ) -> GridFilterFit:
        return GridFilterFit(
            self,
            y,
            params,
            p_0,
            mle_method=mle_method,
            mle_n_iter=mle_n_iter,
            mle_return_chain=mle_return_chain,
            mle_kwargs=mle_kwargs or {},
            t=t,
            param_names=param_names or self.param_names,
        )


__getattr__ = deprecated_aliases(
    __name__,
    {
        "General1DStateSpaceModelFit": "GridFilterFit",
        "General1DStateSpaceModel": "GridFilterModel",
    },
)
