from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...lib.linear_gaussian.model.local_level import forecast as _forecast
from ...lib.linear_gaussian.model.local_level import kalman_filtering as _kalman_filtering
from ...lib.linear_gaussian.model.local_level import kalman_smoothing as _kalman_smoothing
from ...lib.linear_gaussian.model.local_level import smoothed_state_to_obs as _smoothed_state_to_obs
from .. import GeneralLinearGaussianStateSpaceModelFit, GenericLinearGaussianStateSpaceModel

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt

    from .. import Forecast, KalmanFiltering, KalmanSmoothing, SmoothedStateToObs


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
    def smoothed_state_to_obs(self) -> SmoothedStateToObs:
        return _smoothed_state_to_obs

    @property
    def forecast(self) -> Forecast:
        return _forecast


local_level_model = LocalLevelModel()
