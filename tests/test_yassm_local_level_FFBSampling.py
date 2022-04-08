from __future__ import annotations

from unittest import TestCase

import numpy as np

from yassm.linear_gaussian.model.local_level import local_level_model

from tests import y


class TestLocalLinearGaussian(TestCase):
    y = y

    def setUp(self) -> None:
        self.atol = 2e-2
        mle_n_iter = 10
        n_sample = 10000

        μ_0 = 0.0
        Γ_0 = 100.0  # np.infty

        self.fit = local_level_model.fit(
            self.y,
            None,
            μ_0,
            Γ_0,
            mle_n_iter=mle_n_iter,
            n_sample=n_sample,
        )

    def test_FFBSampling(self) -> None:
        fit = self.fit
        np.random.seed(11111)
        np.testing.assert_allclose(fit.m_t_given_T_from_samples, fit.m_t_given_T, atol=self.atol)
        np.testing.assert_allclose(fit.Q_t_given_T_from_samples, fit.Q_t_given_T, atol=self.atol)
