from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import TestCase

import numpy as np
from numba import jit

from tests import set_seed, y
from yassm.linear_gaussian.model.local_level import local_level_model
from yassm.particle_filter import LinearGaussianParticleFilterModel

if TYPE_CHECKING:
    import numpy.typing as npt

y_max = y.max()
y_min = y.min()
y_mid = 0.5 * (y_max + y_min)
y_rge = 0.5 * (y_max - y_min)
μ_0 = y_mid
Γ_0 = y_rge * y_rge


@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def A0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.ones((1, 1))


@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def B0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.ones((1, 1))


# state cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def Σ0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    res = np.empty((1, 1))
    res[0, 0] = params[1]
    return res


# obs. cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def R0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    res = np.empty((1, 1))
    res[0, 0] = params[0]
    return res


@jit("Tuple((f8[::1], f8[:, ::1]))(f8[::1])", nopython=True, nogil=True, cache=True)
def X0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    temp = np.empty((2, 1))
    μ = temp[0]
    Γ = temp[1].reshape((1, 1))
    μ[0] = μ_0
    Γ[0, 0] = Γ_0
    return μ, Γ


class TestLinearGaussianParticleFilterModelBootstrap(TestCase):
    def setUp(self) -> None:
        set_seed(20220501)
        n_sample = 5000
        self.rtol = 0.1
        # test starting from this t-step
        # as particle filter can be inaccurate in early t.
        self.t_min = 100
        self.fit = local_level_model.fit(y, None, μ_0, Γ_0, mle_n_iter=100)

        self.params = self.fit.params

        self.bootstrap_local_level_model = LinearGaussianParticleFilterModel(
            2,
            A0,
            B0,
            Σ0,
            R0,
            X0=X0,
            param_names=["obs_cov", "state_cov"],
            particle_filter_method="bootstrap",
        )
        self.bootstrap_local_level_model.params_guess = local_level_model.params_guess

        self.fit_bootstrap = self.bootstrap_local_level_model.fit(
            y,
            self.params,
            n_sample=n_sample,
            param_names=local_level_model.param_names,
        )

        self.optimal_local_level_model = LinearGaussianParticleFilterModel(
            2,
            A0,
            B0,
            Σ0,
            R0,
            X0=X0,
            param_names=["obs_cov", "state_cov"],
            particle_filter_method="optimal",
        )
        self.optimal_local_level_model.params_guess = local_level_model.params_guess

        self.fit_optimal = self.optimal_local_level_model.fit(
            y,
            self.params,
            n_sample=n_sample,
            param_names=local_level_model.param_names,
        )

    def test_filter_bootstrap(self):
        np.testing.assert_allclose(
            self.fit_bootstrap.m_t_given_t[self.t_min :], self.fit.m_t_given_t[self.t_min :], rtol=self.rtol
        )

    def test_filter_optimal(self):
        np.testing.assert_allclose(
            self.fit_optimal.m_t_given_t[self.t_min :], self.fit.m_t_given_t[self.t_min :], rtol=self.rtol
        )

    def test_log_likelihood(self):
        """The particle filters estimate the full likelihood: the Kalman one plus log p(y_1)."""
        σ_ε_sq = self.params[0]
        log_p_y1 = -0.5 * (np.log(2.0 * np.pi * (Γ_0 + σ_ε_sq)) + (y[0] - μ_0) ** 2 / (Γ_0 + σ_ε_sq))
        expected = self.fit.log_likelihood + log_p_y1
        # Monte Carlo error and bias are below 1 with 5000 particles;
        # dropping log p(y_1) from the optimal filter made it about 7 off
        np.testing.assert_allclose(self.fit_optimal.log_likelihood, expected, atol=2.0)
        np.testing.assert_allclose(self.fit_bootstrap.log_likelihood, expected, atol=2.0)

    def test_grid_search_without_params(self):
        params_grid = np.ascontiguousarray(self.params * np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5]]))
        model = LinearGaussianParticleFilterModel(
            2, A0, B0, Σ0, R0, X0=X0, param_names=["obs_cov", "state_cov"], particle_filter_method="optimal"
        )
        fit = model.fit(
            y, None, mle_method="grid_search", mle_n_iter=1, mle_kwargs={"params_grid": params_grid}, n_sample=500
        )
        np.testing.assert_array_equal(fit.params, self.params)
