"""Define local ratio model using general-linear-gaussian framework,
and then the local ratio model against it.
"""

from __future__ import annotations

from unittest import TestCase

import numpy as np

from tests import y
from yassm.linear_gaussian.model.local_ratio import local_ratio_model
from yassm.linear_gaussian.model.local_ratio_general import general_local_ratio_model


class TestLocalRatioGeneral(TestCase):
    y = y
    μ_0 = np.array([0.0])
    Γ_0 = np.array([[1e5]])
    rtol = 1e-9
    params = np.array([1e-2, 1e-2, 0.5])
    n_sample = 10000

    def setUp(self) -> None:
        self.fit = local_ratio_model.fit(
            self.y,
            self.params,
            self.μ_0,
            self.Γ_0,
            n_sample=self.n_sample,
        )
        self.fit_general = general_local_ratio_model.fit(
            self.y,
            self.params,
            self.μ_0,
            self.Γ_0,
            n_sample=self.n_sample,
        )
        # allow 2σ
        self.rtol_sample_m = 2.0 / np.sqrt(self.n_sample)
        # allow 10σ
        self.rtol_sample_Q = 10.0 * np.sqrt(2.0 / (self.n_sample - 1))
        scale = np.max(np.abs(self.y))
        self.atol_sample_m = scale * 1e-2
        self.atol_sample_Q = 0.0

    def test_kalman_filter_m_t_given_t(self) -> None:
        np.testing.assert_allclose(self.fit.m_t_given_t, self.fit_general.m_t_given_t, rtol=self.rtol)

    def test_kalman_filter_Q_t_given_t(self) -> None:
        np.testing.assert_allclose(self.fit.Q_t_given_t, self.fit_general.Q_t_given_t, rtol=self.rtol)

    def test_kalman_filter_m_t_given_tMinus1(self) -> None:
        np.testing.assert_allclose(self.fit.m_t_given_tMinus1, self.fit_general.m_t_given_tMinus1, rtol=self.rtol)

    def test_kalman_filter_Q_t_given_tMinus1(self) -> None:
        np.testing.assert_allclose(self.fit.Q_t_given_tMinus1, self.fit_general.Q_t_given_tMinus1, rtol=self.rtol)

    def test_kalman_smoother_m_t_given_T(self) -> None:
        np.testing.assert_allclose(self.fit.m_t_given_T, self.fit_general.m_t_given_T, rtol=self.rtol)

    def test_kalman_smoother_Q_t_given_T(self) -> None:
        np.testing.assert_allclose(self.fit.Q_t_given_T, self.fit_general.Q_t_given_T, rtol=self.rtol)

    def test_FFBSampling_m_t_given_T(self) -> None:
        np.testing.assert_allclose(
            self.fit.m_t_given_T_from_samples, self.fit.m_t_given_T, rtol=self.rtol_sample_m, atol=self.atol_sample_m
        )

    def test_FFBSampling_m_t_given_T_general(self) -> None:
        np.testing.assert_allclose(
            self.fit_general.m_t_given_T_from_samples,
            self.fit_general.m_t_given_T,
            rtol=self.rtol_sample_m,
            atol=self.atol_sample_m,
        )

    def test_FFBSampling_Q_t_given_T(self) -> None:
        np.testing.assert_allclose(
            self.fit.Q_t_given_T_from_samples, self.fit.Q_t_given_T, rtol=self.rtol_sample_Q, atol=self.atol_sample_Q
        )

    def test_FFBSampling_Q_t_given_T_general(self) -> None:
        np.testing.assert_allclose(
            self.fit_general.Q_t_given_T_from_samples,
            self.fit_general.Q_t_given_T,
            rtol=self.rtol_sample_Q,
            atol=self.atol_sample_Q,
        )


class TestLocalRatioGeneral2(TestLocalRatioGeneral):
    params = np.array([y.std(), y.std(), 1.0])


class TestLocalRatioGeneral3(TestLocalRatioGeneral):
    params = np.array([y.std(), 1e-5, 0.1])


class TestLocalRatioGeneral4(TestLocalRatioGeneral):
    params = np.array([y.std(), 1e-1, -0.1])


class TestLocalRatioGeneral5(TestLocalRatioGeneral):
    params = np.array([y.std(), y.std(), -1.0])


class TestLocalRatioGeneralMle(TestCase):
    y = y
    μ_0 = np.array([0.0])
    Γ_0 = np.array([[1e5]])
    rtol = 1e-4

    def setUp(self) -> None:
        self.fit = local_ratio_model.fit(
            self.y,
            None,
            self.μ_0,
            self.Γ_0,
            mle_n_iter=1000,
            n_sample=100,
        )
        self.fit_general = general_local_ratio_model.fit(
            self.y,
            np.array([y.std(), 0.1, 1.0]),
            self.μ_0,
            self.Γ_0,
            mle_n_iter=1,
            n_sample=100,
        )

    def test_mle(self) -> None:
        np.testing.assert_allclose(self.fit.params, self.fit_general.params, rtol=self.rtol)
