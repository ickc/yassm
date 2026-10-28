from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import TestCase

import numpy as np
from numba import jit

from tests import set_seed, y
from yassm.lib.linear_gaussian import prob_normal_1d
from yassm.linear_gaussian.model.local_level import local_level_model
from yassm.particle_filter import NonLinearParticleFilterModel

if TYPE_CHECKING:
    import numpy.typing as npt

y_max = y.max()
y_min = y.min()
y_mid = 0.5 * (y_max + y_min)
y_rge = 0.5 * (y_max - y_min)
μ_0 = y_mid
Γ_0 = y_rge * y_rge
C_sqrt = y_rge


@jit("(f8[::1], f8[::1], f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def pdf_Y_inplace(pdf, y_t, x, params, t):
    σ_ε_sq, σ_Z_sq = params
    pdf[:] = prob_normal_1d(y_t[0], x.flatten(), σ_ε_sq, False)


@jit("(f8[:, ::1], f8[:, ::1], f8[::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def K_t_inplace(x_t, x_tMinus1, u_t, params, t):
    x_t[:] = x_tMinus1 + u_t.reshape(-1, 1)


@jit("(f8[:, ::1], f8[::1])", nopython=True, nogil=True, cache=True)
def sampling_X0_tilde_inplace(x_0_tilde, params):
    N = x_0_tilde.shape[0]
    x_0_tilde[:, 0] = np.random.normal(0.0, C_sqrt, N)


@jit("(f8[::1], i8, i8)", nopython=True, nogil=True, cache=True)
def sampling_U_t(params, t, N):
    σ_ε_sq, σ_Z_sq = params
    return np.random.normal(0.0, np.sqrt(σ_Z_sq), N)


@jit(
    "f8[:, :, ::1](f8[::1], f8[:, :, ::1], int64)",
    nopython=True,
    nogil=True,
    cache=True,
)
def state_to_obs(
    params: npt.NDArray[np.float64],
    x_t: npt.NDArray[np.float64],
    t0: int,
) -> npt.NDArray[np.float64]:
    return x_t


class TestNonLinearParticleFilterModelBootstrap(TestCase):
    def setUp(self) -> None:
        set_seed(20220501)
        n_sample = 5000
        self.rtol = 0.2
        # test starting from this t-step
        # as particle filter can be inaccurate in early t.
        self.t_min = 100
        self.fit = local_level_model.fit(y, None, μ_0, Γ_0, mle_n_iter=100)

        self.params = self.fit.params

        self.bootstrap_local_level_model = NonLinearParticleFilterModel(
            2,
            pdf_Y_inplace=pdf_Y_inplace,
            K_t_inplace=K_t_inplace,
            sampling_X0_tilde_inplace=sampling_X0_tilde_inplace,
            sampling_U_t=sampling_U_t,
            state_to_obs=state_to_obs,
            param_names=["σ_ε_sq", "σ_Z_sq"],
            particle_filter_method="bootstrap",
        )
        self.bootstrap_local_level_model.params_guess = local_level_model.params_guess

        self.fit_bootstrap = self.bootstrap_local_level_model.fit(
            y,
            self.params,
            n_sample=n_sample,
            param_names=local_level_model.param_names,
        )

    def test_filter_bootstrap(self):
        np.testing.assert_allclose(
            self.fit_bootstrap.m_t_given_t[self.t_min :], self.fit.m_t_given_t[self.t_min :], rtol=self.rtol
        )
