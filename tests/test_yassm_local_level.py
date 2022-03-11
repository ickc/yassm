from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import TestCase

import numpy as np
from numba import jit

from yassm.linear_gaussian import GeneralLinearGaussianStateSpaceModel
from yassm.linear_gaussian.model.local_level import local_level_model

if TYPE_CHECKING:
    import numpy.typing as npt


@jit("float64[::1](float64[::1])", nogil=True, nopython=True)
def fn(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.sin(15.0 * x) + 3.0 * np.exp(-0.5 * np.square(x)) + 0.5 * np.square(x - 0.5)


@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def A0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array(
        [
            [1.0],
        ]
    )


@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def B0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([[1.0]])


# state cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def Σ0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([[params[1]]])


# obs. cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def R0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([[params[0]]])


local_level_model_general = GeneralLinearGaussianStateSpaceModel(
    2,
    A0,
    B0,
    Σ0,
    R0,
    param_names=["obs_cov", "state_cov"],
)


class TestLinearGaussian(TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(54321)

        n = 500
        sig = 1.0
        C = 1e8
        self.atol = 1e-5
        self.T_predict = 100

        xx = np.linspace(0.0, 1.0, n)
        yy_0 = fn(xx)
        self.yy = yy_0 + rng.normal(scale=sig, size=n)
        self.y = self.yy.reshape(-1, 1)

        self.μ_0 = np.zeros(1)
        self.Γ_0 = np.array([[C]])

        # we don't care if it is MLE
        self.params0 = np.ones(2)
        self.fit = local_level_model.fit(
            self.y,
            self.params0,
            self.μ_0,
            self.Γ_0,
        )
        self.fit_ref = local_level_model_general.fit(
            self.y,
            self.params0,
            self.μ_0,
            self.Γ_0,
        )

    def test_kalman_filtering(self) -> None:
        np.testing.assert_allclose(
            self.fit.m_t_given_t,
            self.fit_ref.m_t_given_t,
            atol=self.atol,
        )
        np.testing.assert_allclose(
            self.fit.Q_t_given_t,
            self.fit_ref.Q_t_given_t,
            atol=self.atol,
        )
        np.testing.assert_allclose(
            self.fit.m_t_given_tMinus1,
            self.fit_ref.m_t_given_tMinus1,
            atol=self.atol,
        )
        np.testing.assert_allclose(
            self.fit.Q_t_given_tMinus1,
            self.fit_ref.Q_t_given_tMinus1,
            atol=self.atol,
        )

    def test_kalman_smoothing(self) -> None:
        np.testing.assert_allclose(
            self.fit.m_t_given_T,
            self.fit_ref.m_t_given_T,
            atol=self.atol,
        )
        np.testing.assert_allclose(
            self.fit.Q_t_given_T,
            self.fit_ref.Q_t_given_T,
            atol=self.atol,
        )

    def test_forecast(self) -> None:
        forecast, forecast_cov = self.fit.get_forecast_obs(self.T_predict)
        forecast_ref, forecast_cov_ref = self.fit_ref.get_forecast_obs(self.T_predict)

        np.testing.assert_allclose(
            forecast,
            forecast_ref,
            atol=self.atol,
        )
        np.testing.assert_allclose(
            forecast_cov,
            forecast_cov_ref,
            atol=self.atol,
        )
