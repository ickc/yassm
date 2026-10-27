from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from unittest import TestCase

import numpy as np
import statsmodels.tsa.api as tsa

from yassm.linear_gaussian.model.local_ratio import local_ratio_model

from . import y
from .test_yassm_local_ratio_with_sm import LocalRatioModel

if TYPE_CHECKING:
    from typing import Any

    from yassm.linear_gaussian import GeneralLinearGaussianStateSpaceModelFit


class TestLinearGaussianLocalRatioEM(TestCase):
    y = y
    method = "em"
    mle_kwargs: ClassVar[dict[str, Any]] = {}

    def setUp(self) -> None:
        C = 1e8
        self.mle_n_iter = 100
        # the MLE can differs quite a lot unfortunately
        self.rtol = 0.2

        self.μ_0 = np.zeros(1)
        self.Γ_0 = np.array([[C]])

        self.sm = self.setUpSM()
        self.fit = self.setUpYassm()

    def setUpSM(self) -> tsa.statespace.mlemodel.MLEResultsWrapper:
        """Reference answer from statsmodels."""
        mod = LocalRatioModel(self.y, m0=self.μ_0, C0=self.Γ_0)
        return mod.fit()

    def setUpYassm(self) -> GeneralLinearGaussianStateSpaceModelFit:
        # param = np.empty(3)
        # param[:2] = self.y.var()
        # param[2] = 1.0
        param = self.sm.params
        return local_ratio_model.fit(
            self.y,
            param,
            self.μ_0,
            self.Γ_0,
            mle_method=self.method,
            mle_n_iter=self.mle_n_iter,
            mle_kwargs=self.mle_kwargs,
        )

    def test_mle(self) -> None:
        np.testing.assert_allclose(self.fit.params, self.sm.params, rtol=self.rtol)


class TestLinearGaussianLocalRatioScipyMinimize(TestLinearGaussianLocalRatioEM):
    method = "scipy_minimize"
    mle_kwargs: ClassVar[dict[str, Any]] = {"bounds": ((0, None), (0, None), (None, None))}


# failure if learning rate is too high
# not testing this for now
# class TestLinearGaussianLocalRatioGradient(TestLinearGaussianLocalRatioEM):
#     method = "gradient"
