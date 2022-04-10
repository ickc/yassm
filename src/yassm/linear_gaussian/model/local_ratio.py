from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from ...lib.linear_gaussian.model.local_ratio import backward_sampling as _backward_sampling
from ...lib.linear_gaussian.model.local_ratio import forecast as _forecast
from ...lib.linear_gaussian.model.local_ratio import gibbs_sampling as _gibbs_sampling
from ...lib.linear_gaussian.model.local_ratio import kalman_filtering as _kalman_filtering
from ...lib.linear_gaussian.model.local_ratio import kalman_smoothing as _kalman_smoothing
from ...lib.linear_gaussian.model.local_ratio import neg2_log_likelihood as _neg2_log_likelihood
from ...lib.linear_gaussian.model.local_ratio import sampling_parameters as _sampling_parameters
from ...lib.linear_gaussian.model.local_ratio import smoothed_state_to_obs as _smoothed_state_to_obs
from .local_level import LocalLevelModel

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt

    from .. import (
        FFBSampling,
        Forecast,
        GibbsSampling,
        KalmanFiltering,
        KalmanSmoothing,
        Likelihood,
        Mle,
        SamplingParameters,
        SmoothedStateToObs,
    )

logger = getLogger(__name__)


@dataclass
class LocalRatioModel(LocalLevelModel):
    """Local ratio model."""

    N: int = 3

    def __post_init__(self) -> None:
        self.param_names: list[str] | None = ["obs_cov", "state_cov", "φ"]

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

    @cached_property
    def mle_gradient(self) -> Mle:
        from ...lib.linear_gaussian.model.local_ratio import (
            _kalman_filtering_1d_inplace,
            _kalman_smoothing_1d_inplace,
            param_update_inplace_gradient,
        )

        # now we can cache this
        return self.get_mle_func(
            _kalman_filtering_1d_inplace, _kalman_smoothing_1d_inplace, param_update_inplace_gradient
        )

    @cached_property
    def mle_em(self) -> Mle:
        from ...lib.linear_gaussian.model.local_ratio import (
            _kalman_filtering_1d_inplace,
            _kalman_smoothing_1d_inplace,
            param_update_inplace_em,
        )

        # now we can cache this
        return self.get_mle_func(_kalman_filtering_1d_inplace, _kalman_smoothing_1d_inplace, param_update_inplace_em)

    def mle(self, method: str = "gradient") -> Mle:
        method_lower = method.lower()
        if method_lower == "gradient":
            return self.mle_gradient
        elif method_lower == "em":
            return self.mle_em
        elif method_lower == "scipy_minimize":
            return self.mle_scipy_minimize
        else:
            logger.warning("method %s not supported, defaulting to EM", method)
            return self.mle_em

    @staticmethod
    def params_guess(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # guess φ = 1 and then use the same guess from local level model
        # law of large number guesstimate
        # ~ 2 * σ_ε_sq + 2 * σ_Z_sq
        temp1 = np.nanvar(y[2:] - y[:-2])
        # ~ 2 * σ_ε_sq + σ_Z_sq
        temp2 = np.nanvar(np.diff(y))
        σ_Z_sq0 = temp1 - temp2
        σ_ε_sq0 = 0.5 * (temp2 - σ_Z_sq0)
        params0 = np.array([σ_ε_sq0, σ_Z_sq0, 1.0])
        return params0


local_ratio_model = LocalRatioModel()
