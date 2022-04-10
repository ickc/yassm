from __future__ import annotations

from unittest import TestCase

import numpy as np

from yassm.linear_gaussian.model.local_level import local_level_model
from yassm.linear_gaussian.model.local_ratio import local_ratio_model

from tests import set_seed, y


class TestLocalLinearGaussian(TestCase):
    y = y
    model = local_level_model

    def setUp(self) -> None:
        self.mle_n_iter = 100
        self.n_sample = 10000
        self.gibbs_burn_in = 500

        μ_0 = 0.0
        Γ_0 = np.infty

        self.fit = self.model.fit(
            self.y,
            None,
            μ_0,
            Γ_0,
            mle_n_iter=self.mle_n_iter,
            n_sample=self.n_sample,
            gibbs_burn_in=self.gibbs_burn_in,
        )
        # allow 2σ
        self.rtol_sample_m = 2.0 / np.sqrt(self.n_sample)
        # allow 4σ
        self.rtol_sample_Q = 4.0 * np.sqrt(2.0 / (self.n_sample - 1))
        scale = np.max(np.abs(self.y))
        self.atol_sample_m = scale * 1e-4
        self.atol_sample_Q = scale**2 * 1e-3

    def test_FFBSampling_m(self) -> None:
        np.testing.assert_allclose(self.fit.m_t_given_T_from_samples, self.fit.m_t_given_T, rtol=self.rtol_sample_m)

    def test_FFBSampling_Q(self) -> None:
        np.testing.assert_allclose(self.fit.Q_t_given_T_from_samples, self.fit.Q_t_given_T, rtol=self.rtol_sample_Q)

    def test_gibbs_sampling_m(self) -> None:
        np.testing.assert_allclose(
            self.fit.m_t_given_T_from_gibbs_samples,
            self.fit.m_t_given_T,
            rtol=self.rtol_sample_m,
            atol=self.atol_sample_m,
        )

    def test_gibbs_sampling_Q(self) -> None:
        np.testing.assert_allclose(
            self.fit.Q_t_given_T_from_gibbs_samples,
            self.fit.Q_t_given_T,
            rtol=self.rtol_sample_Q,
            atol=self.atol_sample_Q,
        )


class TestLocalRatioGaussian(TestLocalLinearGaussian):
    y = y
    model = local_ratio_model
