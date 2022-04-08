from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...lib.linear_gaussian.model.local_linear import A0, B0, R0, Σ0
from .. import GeneralLinearGaussianStateSpaceModel

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class LocalLinearModel(GeneralLinearGaussianStateSpaceModel):
    @staticmethod
    def params_guess(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # law of large number guesstimate
        diff = y[2:] - 2.0 * y[1:-1] + y[:-2]
        # ~ 6 * σ_ε_sq + σ_Z_sq
        temp1 = np.nanvar(diff)
        # ~ 20 * σ_ε_sq + 2 * σ_Z_sq
        temp2 = np.nanvar(np.diff(diff))
        σ_ε_sq = (temp2 - 2.0 * temp1) / 8.0
        σ_Z_sq = temp1 - 6.0 * σ_ε_sq
        return np.array([σ_ε_sq, σ_Z_sq])


local_linear_model = LocalLinearModel(
    2,
    A0,
    B0,
    Σ0,
    R0,
    # param_names=["σ_ε", "σ_η"],
    param_names=["obs_cov", "state_cov"],
    mle_kwargs={"bounds": ((0.0, None), (0.0, None))},
)
