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

    def fit(
        self,
        y: npt.NDArray[np.float64],
        params: npt.NDArray[np.float64],
        μ_0: npt.NDArray[np.float64],
        Γ_0: npt.NDArray[np.float64],
        t: npt.NDArray[Any] | None = None,
        param_names: list[str] | None = ["obs_cov", "state_cov"],
    ) -> GeneralLinearGaussianStateSpaceModelFit:
        """Compile the model and return a GeneralLinearGaussianStateSpaceModelFit instance.

        :param y: observed data (Y)
        :param params: parameters of the model
        :param μ_0: mean of X0
        :param Γ_0: covariance of X0
        :param t: optional time axis
        :param param_names: optional names of parameters
        """
        return GeneralLinearGaussianStateSpaceModelFit(
            y,
            params,
            μ_0,
            Γ_0,
            self.kalman_filtering,
            self.kalman_smoothing,
            self.smoothed_state_to_obs,
            self.forecast,
            1,
            1,
            self.N,
            t=t,
            param_names=param_names,
        )


local_level_model = LocalLevelModel()
