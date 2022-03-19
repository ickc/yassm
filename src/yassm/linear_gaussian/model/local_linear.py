from ...lib.linear_gaussian.model.local_linear import A0, B0, R0, Σ0
from .. import GeneralLinearGaussianStateSpaceModel

local_linear_model = GeneralLinearGaussianStateSpaceModel(
    2,
    A0,
    B0,
    Σ0,
    R0,
    # param_names=["σ_ε", "σ_η"],
    param_names=["obs_cov", "state_cov"],
    mle_kwargs={"bounds": ((0.0, None), (0.0, None))},
)
