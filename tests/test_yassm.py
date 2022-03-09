import numpy as np
import statsmodels.tsa.api as tsa
from numba import jit

from yassm.linear_gaussian import GeneralLinearGaussianStateSpaceModel

rng = np.random.default_rng(12345)


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


def test_linear_gaussian():
    # statsmodels
    n = 500
    xx = np.linspace(0.0, 1.0, n)
    yy_0 = fn(xx)
    sig = 1.0
    yy = yy_0 + sig * rng.normal(n)

    μ_0 = np.zeros(2)
    Γ_0 = 1e8 * np.identity(2)

    mod = LocalLinearModel(yy, m0=μ_0, C0=Γ_0)
    res = mod.fit()

    # ours
    y = yy
    params0 = res.params

    @jit
    def A(params, t):
        return np.array(
            [
                [2.0, -1.0],
                [1.0, 0.0],
            ]
        )

    @jit
    def B(params, t):
        return np.array([[1.0, 0.0]])

    # state cov.
    @jit
    def Σ(params, t):
        res = np.zeros((2, 2))
        res[0, 0] = params[1]
        return res

    # obs. cov.
    @jit
    def R(params, t):
        return np.array([[params[0]]])

    mod = GeneralLinearGaussianStateSpaceModel(
        y,
        A,
        B,
        Σ,
        R,
        μ_0,
        Γ_0,
        params0,
    )

    m_t_given_tMinus1, Q_t_given_tMinus1, m_t_given_t, Q_t_given_t = mod.kalman_filtering(params0)

    np.testing.assert_allclose(res.filtered_state.transpose(), m_t_given_t, atol=1e-6)
    np.testing.assert_allclose(res.filtered_state_cov.transpose(2, 0, 1), Q_t_given_t, atol=1e-6)
    np.testing.assert_allclose(res.predicted_state.transpose()[:-1], m_t_given_tMinus1, atol=1e-5)
    np.testing.assert_allclose(res.predicted_state_cov.transpose(2, 0, 1)[:-1], Q_t_given_tMinus1, atol=1e-6)
