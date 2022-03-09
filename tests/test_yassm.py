from unittest import TestCase

import numpy as np
import statsmodels.tsa.api as tsa
from numba import jit

from yassm.linear_gaussian import GeneralLinearGaussianStateSpaceModel


@jit("float64[::1](float64[::1])", nogil=True, nopython=True)
def fn(x: np.ndarray) -> np.ndarray:
    return np.sin(15.0 * x) + 3.0 * np.exp(-0.5 * np.square(x)) + 0.5 * np.square(x - 0.5)


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
        endog,
        m0: np.ndarray | None = None,
        C0: np.ndarray | None = None,
        exog=None,
        dates=None,
        freq=None,
        initialization="approximate_diffuse",
        **kwargs,
    ):
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
    def param_names(self):
        return ["obs_cov", "state_cov"]

    @property
    def start_params(self):
        return np.full(2, np.std(self.endog), dtype=self.endog.dtype)

    def transform_params(self, unconstrained):
        return np.square(unconstrained)

    def untransform_params(self, constrained):
        return np.sqrt(constrained)

    def update(
        self,
        params,
        transformed=True,
        includes_fixed=False,
        complex_step=False,
    ):
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


@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True, cache=True)
def A(params, t):
    return np.array(
        [
            [2.0, -1.0],
            [1.0, 0.0],
        ]
    )


@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True, cache=True)
def B(params, t):
    return np.array([[1.0, 0.0]])


# state cov.
@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True, cache=True)
def Σ(params, t):
    res = np.zeros((2, 2))
    res[0, 0] = params[1]
    return res


# obs. cov.
@jit("f8[:, ::1](f8[::1], i8)", nopython=True, nogil=True, cache=True)
def R(params, t):
    return np.array([[params[0]]])


class TestLinearGaussian(TestCase):
    def setUp(self):
        rng = np.random.default_rng(12345)

        n = 500
        sig = 1.0
        C = 1e8
        self.atol = 1e-5

        xx = np.linspace(0.0, 1.0, n)
        yy_0 = fn(xx)
        self.yy = yy_0 + rng.normal(scale=sig, size=n)
        self.y = self.yy.reshape(-1, 1)

        self.μ_0 = np.zeros(2)
        self.Γ_0 = C * np.identity(2)

        self.sm = self.setUpSM()
        self.params0 = self.sm.params
        self.mod = self.setUpLocal()

        (
            self.m_t_given_tMinus1,
            self.Q_t_given_tMinus1,
            self.m_t_given_t,
            self.Q_t_given_t,
        ) = self.mod.kalman_filtering(
            self.y,
            self.params0,
            self.μ_0,
            self.Γ_0,
        )

    def setUpSM(self):
        """Reference answer from statsmodels."""
        mod = LocalLinearModel(self.yy, m0=self.μ_0, C0=self.Γ_0)
        return mod.fit()

    def setUpLocal(self):
        return GeneralLinearGaussianStateSpaceModel(
            A,
            B,
            Σ,
            R,
            2,
        )

    def test_kalman_filtering(self):
        res = self.sm

        np.testing.assert_allclose(res.filtered_state.transpose(), self.m_t_given_t, atol=self.atol)
        np.testing.assert_allclose(res.filtered_state_cov.transpose(2, 0, 1), self.Q_t_given_t, atol=self.atol)
        np.testing.assert_allclose(res.predicted_state.transpose()[:-1], self.m_t_given_tMinus1, atol=self.atol)
        np.testing.assert_allclose(
            res.predicted_state_cov.transpose(2, 0, 1)[:-1], self.Q_t_given_tMinus1, atol=self.atol
        )

    def test_kalman_smoothing(self):
        res = self.sm
        m_t_given_T, Q_t_given_T = self.mod.kalman_smoothing(
            self.y,
            self.params0,
            self.m_t_given_tMinus1,
            self.Q_t_given_tMinus1,
            self.m_t_given_t,
            self.Q_t_given_t,
        )
        np.testing.assert_allclose(res.smoothed_state.transpose(), m_t_given_T, atol=self.atol)
        np.testing.assert_allclose(res.smoothed_state_cov.transpose(2, 0, 1), Q_t_given_T, atol=self.atol)
