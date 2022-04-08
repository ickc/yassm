from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
from numba import jit

from ...lib.linear_gaussian.model.local_level import backward_sampling as _backward_sampling
from ...lib.linear_gaussian.model.local_level import forecast as _forecast
from ...lib.linear_gaussian.model.local_level import gibbs_sampling as _gibbs_sampling
from ...lib.linear_gaussian.model.local_level import kalman_filtering as _kalman_filtering
from ...lib.linear_gaussian.model.local_level import kalman_smoothing as _kalman_smoothing
from ...lib.linear_gaussian.model.local_level import neg2_log_likelihood as _neg2_log_likelihood
from ...lib.linear_gaussian.model.local_level import neg2_log_likelihoods as _neg2_log_likelihoods
from ...lib.linear_gaussian.model.local_level import sampling_parameters as _sampling_parameters
from ...lib.linear_gaussian.model.local_level import smoothed_state_to_obs as _smoothed_state_to_obs
from .. import GenericLinearGaussianStateSpaceModel

if TYPE_CHECKING:
    from typing import Callable

    import numpy.typing as npt

    from .. import (
        FFBSampling,
        Forecast,
        GibbsSampling,
        KalmanFiltering,
        KalmanSmoothing,
        Mle,
        ParamUpdateInplace,
        SamplingParameters,
        SmoothedStateToObs,
    )

    KalmanFilteringInplace1D = Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            float,
            float,
        ],
        None,
    ]
    KalmanSmoothingInplace1D = Callable[
        [
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

logger = getLogger(__name__)


@dataclass
class LocalLevelModel(GenericLinearGaussianStateSpaceModel):
    """Local level model."""

    N: int = 2

    def __post_init__(self) -> None:
        self.param_names: list[str] | None = ["obs_cov", "state_cov"]

        self.d: int = 1
        self.n: int = 1

    @property
    def kalman_filtering(self) -> KalmanFiltering:
        return _kalman_filtering

    @property
    def kalman_smoothing(self) -> KalmanSmoothing:
        return _kalman_smoothing

    @property
    def backward_sampling(self) -> FFBSampling:
        return _backward_sampling

    @property
    def sampling_parameters(self) -> SamplingParameters:
        return _sampling_parameters

    @property
    def gibbs_sampling(self) -> GibbsSampling:
        return _gibbs_sampling

    @property
    def smoothed_state_to_obs(self) -> SmoothedStateToObs:
        return _smoothed_state_to_obs

    @property
    def forecast(self) -> Forecast:
        return _forecast

    @property
    def neg2_log_likelihood(self) -> Likelihood:
        """Calculate negative 2 times log-likelihood.

        Y|θ
        """
        return _neg2_log_likelihood

    @property
    def neg2_log_likelihoods(self) -> Likelihoods:
        """Calculate negative 2 times log-likelihood.

        Y|θ
        """
        return _neg2_log_likelihoods

    def get_mle_func(
        self,
        _kalman_filtering_1d_inplace: KalmanFilteringInplace1D,
        _kalman_smoothing_1d_inplace: KalmanSmoothingInplace1D,
        param_update_inplace: ParamUpdateInplace,
    ) -> Mle:
        @jit(
            "Tuple((f8[::1], f8[:, ::1]))(f8[:, ::1], f8[::1], f8[::1], f8[:, ::1], i8, b1)",
            nopython=True,
            nogil=True,
            cache=False,
        )
        def mle(
            y_2d: npt.NDArray[np.float64],
            params0: npt.NDArray[np.float64],
            μ_0_1d: npt.NDArray[np.float64],
            Γ_0_2d: npt.NDArray[np.float64],
            n_iter: int,
            return_chain: bool,
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            # to 1d
            y = y_2d.flatten()
            μ_0 = μ_0_1d[0]
            Γ_0 = Γ_0_2d[0, 0]

            size = y.size
            N = params0.size

            chain = np.empty((n_iter + 1 if return_chain else 0, N))

            # initialize & memalloc
            params = params0.copy()
            if return_chain:
                chain[0] = params0
            temp = np.empty((6, size))
            m_t_given_tMinus1 = temp[0]
            Q_t_given_tMinus1 = temp[1]
            m_t_given_t = temp[2]
            Q_t_given_t = temp[3]
            m_t_given_T = temp[4]
            Q_t_given_T = temp[5]

            for i in range(1, n_iter + 1):
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
                _kalman_smoothing_1d_inplace(
                    m_t_given_T,
                    Q_t_given_T,
                    params,
                    m_t_given_t,
                    Q_t_given_t,
                )

                param_update_inplace(
                    params,
                    m_t_given_tMinus1,
                    Q_t_given_tMinus1,
                    m_t_given_t,
                    Q_t_given_t,
                    m_t_given_T,
                    Q_t_given_T,
                    y,
                )
                if return_chain:
                    chain[i] = params
            return params, chain

        return mle

    @cached_property
    def mle_em(self) -> Mle:
        from ...lib.linear_gaussian.model.local_level import (
            _kalman_filtering_1d_inplace,
            _kalman_smoothing_1d_inplace,
            param_update_inplace_em,
        )

        # now we can cache this
        return self.get_mle_func(_kalman_filtering_1d_inplace, _kalman_smoothing_1d_inplace, param_update_inplace_em)

    def mle(self, method: str = "gradient") -> Mle:
        method_lower = method.lower()
        if method_lower != "em":
            logger.warning("The only implemented MLE method is EM for local level model. Using EM...")
        return self.mle_em

    @staticmethod
    def params_guess(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # law of large number guesstimate
        # ~ 2 * σ_ε_sq + 2 * σ_Z_sq
        temp1 = np.nanvar(y[2:] - y[:-2])
        # ~ 2 * σ_ε_sq + σ_Z_sq
        temp2 = np.nanvar(np.diff(y))
        σ_Z_sq0 = temp1 - temp2
        σ_ε_sq0 = 0.5 * (temp2 - σ_Z_sq0)
        params0 = np.array([σ_ε_sq0, σ_Z_sq0])
        return params0


local_level_model = LocalLevelModel()
