# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # HW 3: Problem 5
#
# *Originally written on 2022-03-16.*

# %%
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import statsmodels.tsa.api as tsa
from numba import jit
from stat248_data import load_Prob5HW3DataStat248

if TYPE_CHECKING:
    import numpy.typing as npt

# %%
df = load_Prob5HW3DataStat248()
y = df.y.values
n = y.size
t = np.arange(n) / n

# %%
df

# %%
df.plot()


# %% [markdown]
# ## Part a
#
# We'll use `statsmodels` to estimate MLE.


# %%
class Homework3Problem5Model(tsa.statespace.MLEModel):
    """STAT248 HW3 Q5 model."""

    # statsmodels/STAT248 notations
    # Z/B
    _design = np.array([[1.0, 0.0, 0.0, 1.0, 0.0]])
    # R
    _selection = np.array(
        (
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ),
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
        k_states = 5
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
        # T/A
        self.ssm["transition", 0, 0] = 2.0
        self.ssm["transition", 0, 1] = -1.0
        self.ssm["transition", 1, 0] = 1.0
        self.ssm["transition", 2, 1] = 1.0
        self.ssm["transition", 4, 3] = 1.0
        # self.ssm["state_intercept"]
        self.ssm["selection"] = self._selection
        # self.ssm["state_cov"]

    @property
    def param_names(self) -> list[str]:
        return ["obs_cov", "state_cov1", "state_cov2", "φ"]

    @property
    def start_params(self) -> npt.NDArray[np.float64]:
        res = np.full(4, np.std(self.endog), dtype=self.endog.dtype)
        res[3] = 1.0
        return res

    def transform_params(self, unconstrained: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        res = np.empty_like(unconstrained)
        res[:3] = np.square(unconstrained[:3])
        res[3] = unconstrained[3]
        return res

    def untransform_params(self, constrained: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        res = np.empty_like(constrained)
        res[:3] = np.sqrt(constrained[:3])
        res[3] = constrained[3]
        return res

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
        self.ssm["state_cov", 1, 1] = params[2]

        # transition
        self.ssm["transition", 3, 3] = params[3]


# %%
C = 1e8
μ_0 = np.zeros(5)
Γ_0 = np.identity(5) * C

# %%
mod = Homework3Problem5Model(y, m0=μ_0, C0=Γ_0)
res_sm = mod.fit()

# %%
σ_ε_sq, σ_1_sq, σ_2_sq, φ_hat = res_sm.params
σ_ε, σ_1, σ_2, _ = np.sqrt(res_sm.params)
σ_ε, σ_1, σ_2, φ_hat


# %%
pd.Series({"σ̂_ε": σ_ε, "σ̂_1": σ_1, "σ̂_2": σ_2, "φ̂": φ_hat}, name="MLE")

# %% [markdown]
# ## Part b
#
# We'll use my library now.


# %%
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def A0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array(
        [
            [2.0, -1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, params[3], 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )


@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def B0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([[1.0, 0.0, 0.0, 1.0, 0.0]])


# state cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def Σ0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out = np.zeros((5, 5))
    out[0, 0] = params[1]
    out[3, 3] = params[2]
    return out


# obs. cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def R0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out = np.empty((1, 1))
    out[0, 0] = params[0]
    return out


@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def B0_forecast(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])


# %%
from yassm.linear_gaussian import GeneralLinearGaussianStateSpaceModel

# %%
hw3_q5_model = GeneralLinearGaussianStateSpaceModel(
    4,
    A0,
    B0,
    Σ0,
    R0,
    B0_forecast=B0_forecast,
    param_names=["obs_cov", "state_cov1", "state_cov2", "φ"],
)

# %%
res = hw3_q5_model.fit(y, res_sm.params, μ_0, Γ_0, t=t)

# %%
res.plot()

# %%
fig, axes = res.plot(plot_smoothed_CI_with_obs_cov=False, plot_obs=False)
axes[0].set_title("Point estimates of f w/ 95% CI")

# %% [markdown]
# Comparing answer with `statsmodels`,

# %%
atol = 1e-6
rtol = 1e-2

# %%
np.testing.assert_allclose(
    res.m_t_given_t,
    res_sm.filtered_state.transpose(),
    atol=atol,
    rtol=rtol,
)

# %%
np.testing.assert_allclose(
    res.Q_t_given_t,
    res_sm.filtered_state_cov.transpose(2, 0, 1),
    atol=atol,
    rtol=rtol,
)

# %%
np.testing.assert_allclose(
    res.m_t_given_tMinus1,
    res_sm.predicted_state.transpose()[:-1],
    atol=atol,
    rtol=rtol,
)

# %%
np.testing.assert_allclose(
    res.Q_t_given_tMinus1,
    res_sm.predicted_state_cov.transpose(2, 0, 1)[:-1],
    atol=atol,
    rtol=rtol,
)

# %%
np.testing.assert_allclose(
    res.m_t_given_T,
    res_sm.smoothed_state.transpose(),
    atol=atol,
    rtol=rtol,
)

# %%
atol = 0.1
rtol = 0.2

# %%
# statsmodels >= 0.15 differs from statsmodels 0.13 (and from yassm) at t = 1
# under this approximate diffuse prior, so compare from t = 2 onward.
np.testing.assert_allclose(
    res.Q_t_given_T[2:],
    res_sm.smoothed_state_cov.transpose(2, 0, 1)[2:],
    atol=atol,
    rtol=rtol,
)


# %% [markdown]
# ## Part c


# %%
@jit("f8[::1](f8[::1])", nopython=True, nogil=True, cache=False)
def hw3_q4_truth(u):
    return np.sin(15.0 * u) + np.exp(-0.5 * np.square(u)) + 0.5 * np.square(u - 0.5)


# %%
y_0 = hw3_q4_truth(t)

# %%
axes[0].plot(t, y_0, label="original")
axes[0].legend()
fig

# %% [markdown]
# Note that the CI may not be very clear, but the transparent band corresponds to the $95\%$ intervals.
