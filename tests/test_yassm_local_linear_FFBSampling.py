from __future__ import annotations

from unittest import TestCase

import numpy as np

from yassm.linear_gaussian.model.local_linear import local_linear_model

from tests import y


class TestLocalLinearGaussian(TestCase):
    y = y

    def setUp(self) -> None:
        y = self.y

        C = 1e8
        self.atol = 2e-2
        n_sample = 1000

        self.μ_0 = np.zeros(2)
        self.Γ_0 = C * np.identity(2)

        # law of large number guesstimate
        diff = y[2:] - 2.0 * y[1:-1] + y[:-2]
        # ~ 6 * σ_ε_sq + σ_Z_sq
        temp1 = np.nanvar(diff)
        # ~ 20 * σ_ε_sq + 2 * σ_Z_sq
        temp2 = np.nanvar(np.diff(diff))
        σ_ε_sq = (temp2 - 2.0 * temp1) / 8.0
        σ_Z_sq = temp1 - 6.0 * σ_ε_sq
        params0 = np.array([σ_ε_sq, σ_Z_sq])
        del diff, temp1, temp2, σ_ε_sq, σ_Z_sq
        self.params0 = params0

        self.fit = local_linear_model.fit(
            self.y,
            self.params0,
            self.μ_0,
            self.Γ_0,
            mle_n_iter=1,
            n_sample=n_sample,
        )

    def test_FFBSampling(self) -> None:
        fit = self.fit
        np.random.seed(22222)
        np.testing.assert_allclose(fit.m_t_given_T_from_samples, fit.m_t_given_T, atol=self.atol)
        np.testing.assert_allclose(fit.Q_t_given_T_from_samples, fit.Q_t_given_T, atol=self.atol)
