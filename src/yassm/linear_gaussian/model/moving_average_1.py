from __future__ import annotations

from ...lib.linear_gaussian.model.moving_average_1 import A0, B0, R0, X0, Σ0
from .. import LinearGaussianModel

moving_average_1_model = LinearGaussianModel(
    2,
    A0,
    B0,
    Σ0,
    R0,
    X0=X0,
    param_names=["σ_sq", "θ"],
    mle_kwargs={"bounds": ((0.0, None), (None, None))},
)
