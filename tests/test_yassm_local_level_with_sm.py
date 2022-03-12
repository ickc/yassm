from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import TestCase

import numpy as np
import statsmodels.tsa.api as tsa

from yassm.linear_gaussian.model.local_level import local_level_model

from tests import y

if TYPE_CHECKING:
    from yassm.linear_gaussian import GeneralLinearGaussianStateSpaceModelFit


class TestLinearGaussian(TestCase):
    def setUp(self) -> None:
        self.y = y

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
        mod = tsa.UnobservedComponents(self.y, "local level")
        return mod.fit()

    def setUpYassm(self) -> GeneralLinearGaussianStateSpaceModelFit:
        return local_level_model.fit(
            self.y,
            np.full(2, self.y.var()),
            self.μ_0,
            self.Γ_0,
            mle_n_iter=self.mle_n_iter,
        )

    def test_mle(self) -> None:
        np.testing.assert_allclose(self.fit.params, self.sm.params, rtol=self.rtol)
