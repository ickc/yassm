from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import TestCase

import numpy as np
from numba import jit

from tests import set_seed, y
from yassm.lib.linear_gaussian.model.local_linear import A0, B0, R0, Σ0
from yassm.linear_gaussian.model.local_linear import local_linear_model
from yassm.particle_filter import LinearGaussianParticleFilterModel

if TYPE_CHECKING:
    import numpy.typing as npt

y_max = y.max()
y_min = y.min()
y_mid = 0.5 * (y_max + y_min)
y_rge = 0.5 * (y_max - y_min)
μ_0 = np.full(2, y_mid)
Γ_0 = (y_rge * y_rge) * np.identity(2)


@jit(nopython=True, nogil=True, cache=True)
def X0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return μ_0, Γ_0


class TestLinearGaussianParticleFilterModelBootstrap(TestCase):
    def setUp(self) -> None:
        set_seed(20220501)
        n_sample = 5000
        self.rtol = 0.1
        # test starting from this t-step
        # as particle filter can be inaccurate in early t.
        self.t_min = 100
        self.fit = local_linear_model.fit(y, None, μ_0, Γ_0, mle_n_iter=1)

        self.params = self.fit.params

        self.bootstrap_local_linear_model = LinearGaussianParticleFilterModel(
            2,
            A0,
            B0,
            Σ0,
            R0,
            X0=X0,
            param_names=local_linear_model.param_names,
            particle_filter_method="bootstrap",
        )
        self.bootstrap_local_linear_model.params_guess = local_linear_model.params_guess

        self.fit_bootstrap = self.fit_bootstrap = self.bootstrap_local_linear_model.fit(
            y,
            self.params,
            n_sample=n_sample,
        )

        self.optimal_local_linear_model = LinearGaussianParticleFilterModel(
            2,
            A0,
            B0,
            Σ0,
            R0,
            X0=X0,
            param_names=local_linear_model.param_names,
            particle_filter_method="optimal",
        )
        self.optimal_local_linear_model.params_guess = local_linear_model.params_guess

        self.fit_optimal = self.optimal_local_linear_model.fit(
            y,
            self.params,
            n_sample=n_sample,
        )

    def test_filter_bootstrap(self):
        np.testing.assert_allclose(
            self.fit_bootstrap.m_t_given_t[self.t_min :],
            self.fit.m_t_given_t[self.t_min :],
            rtol=self.rtol,
        )

    def test_filter_optimal(self):
        np.testing.assert_allclose(
            self.fit_optimal.m_t_given_t[self.t_min :],
            self.fit.m_t_given_t[self.t_min :],
            rtol=self.rtol,
        )
