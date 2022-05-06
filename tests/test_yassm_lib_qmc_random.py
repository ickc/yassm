from __future__ import annotations

import numpy as np
import pytest

from yassm.lib.qmc.random import (
    correlation_variance_decomposition,
    generate_positive_semidefinite_matrix,
    midpoint_pointset,
    van_der_corput_sequence,
    van_der_corput_sequence_slow,
)
from yassm.qmc.random import gaussian_cholesky_decomposition, gaussian_rosenblatt_transform, guassian_ZCA_cor


@pytest.mark.parametrize(
    "N",
    (3, 5, 11, 31),
)
def test_midpoint_pointset(N) -> None:
    np.testing.assert_allclose(midpoint_pointset(N) * (2 * N), np.arange(1, 2 * N + 1, 2))


@pytest.mark.parametrize(
    "N, base",
    (
        (1000, 2),
        (1000, 3),
        (1000, 11),
        (1000, 31),
    ),
)
def test_van_der_corput_sequence(N, base) -> None:
    np.testing.assert_array_equal(
        van_der_corput_sequence(N, base),
        van_der_corput_sequence_slow(N, base),
    )


GAUSSIAN_CASES = (
    (10000, 2, np.array([[4.0, 1.0], [1.0, 2.0]])),
    (10000, 3, np.array([[4.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 5.0]])),
)


@pytest.mark.parametrize("N, d, cov", GAUSSIAN_CASES)
def test_gaussian_rosenblatt_transform(N, d, cov) -> None:
    uniform_array = np.random.uniform(size=(N, d))
    gaussian = gaussian_rosenblatt_transform(uniform_array, cov)
    cov_sample = np.cov(gaussian, rowvar=False)
    rtol = 5.0 * np.sqrt(2.0 / (N - 1))
    np.testing.assert_allclose(cov_sample, cov, rtol=rtol)


@pytest.mark.parametrize("N, d, cov", GAUSSIAN_CASES)
def test_gaussian_cholesky_decomposition(N, d, cov) -> None:
    uniform_array = np.random.uniform(size=(N, d))
    gaussian = gaussian_cholesky_decomposition(uniform_array, cov)
    cov_sample = np.cov(gaussian, rowvar=False)
    rtol = 5.0 * np.sqrt(2.0 / (N - 1))
    np.testing.assert_allclose(cov_sample, cov, rtol=rtol)


@pytest.mark.parametrize("N, d, cov", GAUSSIAN_CASES)
def test_guassian_ZCA_cor(N, d, cov) -> None:
    uniform_array = np.random.uniform(size=(N, d))
    gaussian = guassian_ZCA_cor(uniform_array, cov)
    cov_sample = np.cov(gaussian, rowvar=False)
    rtol = 5.0 * np.sqrt(2.0 / (N - 1))
    np.testing.assert_allclose(cov_sample, cov, rtol=rtol)


@pytest.mark.parametrize("low, high, size", ((0.0, 1.0, 10), (1.0, 10.0, 10), (-1.0, 1.0, 10)))
def test_correlation_variance_decomposition(low, high, size) -> None:
    Σ = generate_positive_semidefinite_matrix(low, high, size)
    σ, P = correlation_variance_decomposition(Σ)
    np.testing.assert_allclose(σ.reshape(-1, 1) * P * σ, Σ)
