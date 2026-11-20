from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import TestCase

import numpy as np
import statsmodels.tsa.api as tsa

from tests import y
from yassm.linear_gaussian.model.local_linear import local_linear_model

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt


class LocalLinearModel(tsa.statespace.MLEModel):
    """Local linear model."""

    _design = np.array([[1.0, 0.0]])
    _transition = np.array(
        [
            [2.0, -1.0],
            [1.0, 0.0],
        ]
    )

    def __init__(
        self,
        endog: npt.NDArray[np.float64],
        m0: npt.NDArray[np.float64] | None = None,
        C0: npt.NDArray[np.float64] | None = None,
        exog: npt.NDArray[np.float64] | None = None,
        dates: npt.NDArray[Any] | None = None,
        freq: Any = None,
        initialization: str = "approximate_diffuse",
        **kwargs: dict[Any, Any],
    ) -> None:
        k_states = 2
        k_posdef = 2

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            exog=exog,
            dates=dates,
            freq=freq,
            initialization=initialization,
            loglikelihood_burn=k_states,
            **kwargs,
        )
        if m0 is not None and C0 is not None:
            self.ssm.initialize_known(m0, C0)

        # Initialize the matrices
        self.ssm["design"] = self._design
        # self.ssm["obs_intercept"]
        # self.ssm["obs_cov"]
        self.ssm["transition"] = self._transition
        # self.ssm["state_intercept"]
        self.ssm["selection"] = np.eye(k_states)
        # self.ssm["state_cov"]

    @property
    def param_names(self) -> list[str]:
        return ["obs_cov", "state_cov"]

    @property
    def start_params(self) -> npt.NDArray[np.float64]:
        return np.full(2, np.std(self.endog), dtype=self.endog.dtype)

    def transform_params(self, unconstrained: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.square(unconstrained)

    def untransform_params(self, constrained: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.sqrt(constrained)

    def update(
        self,
        params: npt.NDArray[np.float64],
        transformed: bool = True,
        includes_fixed: bool = False,
        complex_step: bool = False,
    ) -> None:
        params = super().update(
            params,
            transformed=True,
            includes_fixed=False,
            complex_step=False,
        )

        # Observation covariance
        self.ssm["obs_cov", 0, 0] = params[0]

        # State covariance
        self.ssm["state_cov", 0, 0] = params[1]


class TestLocalLinearGaussian(TestCase):
    y = y

    def setUp(self) -> None:
        C = 1e8
        self.rtol = 1e-5
        self.T_predict = 100

        self.μ_0 = np.zeros(2)
        self.Γ_0 = C * np.identity(2)

        self.sm = self.setUpSM()
        self.params0 = self.sm.params
        self.fit = local_linear_model.fit(
            self.y,
            self.params0,
            self.μ_0,
            self.Γ_0,
        )
        self.fit_mle = local_linear_model.fit(
            self.y,
            self.params0,
            self.μ_0,
            self.Γ_0,
            mle_n_iter=1,
        )

    def setUpSM(self) -> tsa.statespace.mlemodel.MLEResultsWrapper:
        """Reference answer from statsmodels."""
        mod = LocalLinearModel(self.y, m0=self.μ_0, C0=self.Γ_0)
        return mod.fit()

    def test_kalman_filtering(self) -> None:
        res = self.sm

        np.testing.assert_allclose(
            self.fit.m_t_given_t,
            res.filtered_state.transpose(),
            rtol=self.rtol,
        )
        np.testing.assert_allclose(
            self.fit.Q_t_given_t,
            res.filtered_state_cov.transpose(2, 0, 1),
            rtol=self.rtol,
        )
        np.testing.assert_allclose(
            self.fit.m_t_given_tMinus1,
            res.predicted_state.transpose()[:-1],
            rtol=self.rtol,
        )
        np.testing.assert_allclose(
            self.fit.Q_t_given_tMinus1,
            res.predicted_state_cov.transpose(2, 0, 1)[:-1],
            rtol=self.rtol,
        )

    def test_kalman_smoothing(self) -> None:
        res = self.sm
        np.testing.assert_allclose(
            self.fit.m_t_given_T,
            res.smoothed_state.transpose(),
            rtol=self.rtol,
        )
        np.testing.assert_allclose(
            self.fit.Q_t_given_T,
            res.smoothed_state_cov.transpose(2, 0, 1),
            rtol=self.rtol,
        )

    def test_forecast(self) -> None:
        forecast_sm = self.sm.get_forecast(self.T_predict)
        forecast, forecast_cov = self.fit.get_forecast_obs(self.T_predict)
        np.testing.assert_allclose(
            forecast[:, 0],
            forecast_sm.predicted_mean,
            rtol=self.rtol,
        )
        np.testing.assert_allclose(
            np.sqrt(forecast_cov[:, 0, 0]),
            forecast_sm.se_mean,
            rtol=self.rtol,
        )

    def test_mle(self) -> None:
        np.testing.assert_allclose(self.fit_mle.params, self.sm.params, rtol=0.005)
