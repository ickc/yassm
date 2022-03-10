from ...lib.linear_gaussian.model.local_linear import A0, B0, R0, Σ0, R, Σ
from .. import GeneralLinearGaussianStateSpaceModel

local_linear_model = GeneralLinearGaussianStateSpaceModel(
    A0,
    B0,
    Σ0,
    R0,
    None,
    None,
    Σ,
    R,
)
