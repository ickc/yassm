from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from ...lib.linear_gaussian.model.local_ratio import forecast as _forecast
from ...lib.linear_gaussian.model.local_ratio import kalman_filtering as _kalman_filtering
from ...lib.linear_gaussian.model.local_ratio import kalman_smoothing as _kalman_smoothing
from ...lib.linear_gaussian.model.local_ratio import smoothed_state_to_obs as _smoothed_state_to_obs
from .local_level import LocalLevelModel

if TYPE_CHECKING:
    from .. import Forecast, KalmanFiltering, KalmanSmoothing, Mle, SmoothedStateToObs


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
    def smoothed_state_to_obs(self) -> SmoothedStateToObs:
        return _smoothed_state_to_obs

    @property
    def forecast(self) -> Forecast:
        return _forecast

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
        else:
            raise ValueError


local_ratio_model = LocalRatioModel()
