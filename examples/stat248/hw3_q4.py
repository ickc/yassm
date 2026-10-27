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

# %% [markdown] tags=[]
# # HW 3: Problem 4
#
# *Originally written on 2022-03-16.*

# %%
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import statsmodels.tsa.api as tsa
from numba import jit
from stat248_data import load_Prob4HW3DataStat248

if TYPE_CHECKING:
    import numpy.typing as npt

# %%
df = load_Prob4HW3DataStat248()
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


# %%
mod = LocalLinearModel(df)
res_sm = mod.fit()

# %%
df_hw3_q4a = res_sm.params.to_frame("MLE").apply(np.sqrt)
df_hw3_q4a.index = ["σ_ε", "σ_Z"]
df_hw3_q4a


# %%
σ_ε, σ_Z = df_hw3_q4a.values.flatten()
pd.Series({"σ̂_ε": σ_ε, "σ̂_Z": σ_Z}, name="MLE")

# %% [markdown]
# ## Part b
#
# Now we'll use my own library.

# %%
from yassm.linear_gaussian.model.local_linear import local_linear_model

# %%
res = local_linear_model.fit(y, res_sm.params.values, np.zeros(2), 1e8 * np.identity(2), t=t)

# %%
res.plot()

# %%
fig, axes = res.plot(plot_smoothed_CI_with_obs_cov=False, plot_obs=False)
axes[0].set_title("Point estimates of f w/ 95% CI")


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
