"""General 1-dimensional state space model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
from numba import jit

from ..lib.grid.one_dimension import (
    filter_update_inplace,
    get_grid_to_position,
    one_step_ahead_inplace,
    smoothing_backward_recursion_inplace,
    ps_to_mQ,
)

if TYPE_CHECKING:
    from typing import Any, Callable

    import numpy.typing as npt

logger = getLogger(__name__)


@dataclass
class General1DStateSpaceModelFit:

    parent: General1DStateSpaceModel
    y: npt.NDArray[np.float64]
    params: npt.NDArray[np.float64]
    p_0: npt.NDArray[np.float64]
    t: npt.NDArray[Any] | None = None
    param_names: list[str] | None = None

    def __post_init__(self):
        self.size = self.y.size
        self.G = self.p_0.size
        self.N = self.params.size

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

    def _run_p_t_given_T_to_mQ(self):
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

@dataclass
class General1DStateSpaceModel:
    N: int
    transition_func: Callable[[float, float, npt.NDArray[np.float64]], float]
    design_func: Callable[[float, float, npt.NDArray[np.float64]], float]
    # grid spec
    x_0: float
    x_f: float
    G: int
    param_names: list[str] | None = None

    def __post_init__(self):
        self.grid_to_position = get_grid_to_position(self.x_0, self.x_f, self.G)

    @cached_property
    def kalman_filtering_inplace(self):
        design_func = self.design_func
        grid_to_position = self.grid_to_position
        transition_func = self.transition_func

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
        ):
            """
            :param p_t_given_tMinus1: out array.
            :param p_t_given_t: out array.
            :param y: data.
            :param params: params.
            :param p_0: initial density.
            :return: p_t_given_tMinus1, p_t_given_t
            """
            size = y.shape[0]
            G = p_t_given_tMinus1.size

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
            )
            for t in range(1, size):
                tMinus1 = t - 1
                one_step_ahead_inplace(
                    p_t_given_tMinus1[t],
                    params,
                    transition_func,
                    grid_to_position,
                    p_t_given_t[tMinus1],
                )
                y_t = y[t]
                filter_update_inplace(
                    p_t_given_t[t],
                    y_t,
                    params,
                    design_func,
                    grid_to_position,
                    p_t_given_tMinus1[t],
                )

        return kalman_filtering_inplace

    @cached_property
    def kalman_filtering(self):
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
            temp = np.empty((2, size, G))
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
    def kalman_smoothing_inplace(self):
        transition_func = self.transition_func
        grid_to_position = self.grid_to_position

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
        ):
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
                )

        return kalman_smoothing_inplace

    @cached_property
    def kalman_smoothing(self):
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
            p_t_given_T = np.empty_like(p_t_given_t)

            kalman_smoothing_inplace(
                p_t_given_T,
                params,
                p_t_given_tMinus1,
                p_t_given_t,
            )
            return p_t_given_T

        return kalman_smoothing

    @cached_property
    def smoothed_state_to_obs(self):
        raise NotImplementedError

    @cached_property
    def forecast(self):
        raise NotImplementedError

    def fit(
        self,
        y: npt.ArrayLike,
        params: npt.ArrayLike,
        p_0: npt.ArrayLike,
        t: npt.NDArray[Any] | None = None,
        param_names: list[str] | None = None,
    ):
        return General1DStateSpaceModelFit(
            self,
            y,
            params,
            p_0,
            t=t,
            param_names=param_names or self.param_names,
        )
