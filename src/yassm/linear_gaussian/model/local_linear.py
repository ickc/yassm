from ...lib.linear_gaussian.model.local_linear import A, B, R, Σ
from .. import GeneralLinearGaussianStateSpaceModel

local_linear_model = GeneralLinearGaussianStateSpaceModel(
    A,
    B,
    Σ,
    R,
    2,
)
