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
# # HW 3: Problem 3 (d, e)
#
# *Originally written on 2022-03-15.*

# %% [markdown]
# ## Part d

# %%
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from stat248_data import load_Prob3HW3DataStat248

if TYPE_CHECKING:
    import numpy.typing as npt

# %% [markdown]
# I wrote the `yassm` model for this class. Documentation of `yassm` attached as an appendix.
#
# TODO: made sure I include `yassm` documentation.

# %%
from yassm.linear_gaussian.model.local_ratio import local_ratio_model

# %%
df = load_Prob3HW3DataStat248()

# %%
df

# %%
df.plot()

# %%
y = df.y.values

# %%
μ_0 = np.zeros(1)
Γ_0 = np.identity(1) * np.inf

# %%
# law of large number guesstimate
# ~ 2 * σ_ε_sq + 2 * σ_Z_sq
temp1 = np.nanvar(y[2:] - y[:-2])
# ~ 2 * σ_ε_sq + σ_Z_sq
temp2 = np.nanvar(np.diff(y))
σ_Z_sq0 = temp1 - temp2
σ_ε_sq0 = 0.5 * (temp2 - σ_Z_sq0)
params0 = [σ_ε_sq0, σ_Z_sq0, 1.0]
params0

# %%
# %%time
# the learning rate is hard-coded to be 1e-3 here.
res = local_ratio_model.fit(y, params0, μ_0, Γ_0, mle_method="gradient", mle_n_iter=1000, mle_return_chain=True)

# %%
res.mle_params_chain_to_frame.plot(logx=True, logy=True, title="Iterations by the EM algorithm")


# %%
σ_ε_sq, σ_Z_sq, φ_hat = res.params
pd.Series({"σ̂_ε²": σ_ε_sq, "σ̂_Z²": σ_Z_sq, "φ̂": φ_hat}, name="MLE")

# %% [markdown]
# ## Part e

# %%
# %%time
res_em = local_ratio_model.fit(y, params0, μ_0, Γ_0, mle_method="em", mle_n_iter=1000, mle_return_chain=True)

# %%
res_em.mle_params_chain_to_frame.plot(logx=True, logy=True, title="Iterations by the EM algorithm")

# %%
σ_ε_sq, σ_Z_sq, φ_hat = res_em.params
pd.Series({"σ̂_ε²": σ_ε_sq, "σ̂_Z²": σ_Z_sq, "φ̂": φ_hat}, name="MLE (EM)")

# %%
res_em.plot(100)

# %% [markdown]
# Comparing to `statsmodels`,

# %%
import statsmodels.tsa.api as tsa


class LocalRatioModel(tsa.statespace.MLEModel):
    """Local Ratio model."""

    _design = np.array([[1.0]])

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
        k_states = 1
        k_posdef = 1

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
        # self.ssm["transition"] = self._transition
        # self.ssm["state_intercept"]
        self.ssm["selection"] = np.eye(k_states)
        # self.ssm["state_cov"]

    @property
    def param_names(self) -> list[str]:
        return ["obs_cov", "state_cov", "φ"]

    @property
    def start_params(self) -> npt.NDArray[np.float64]:
        res = np.empty(3, dtype=self.endog.dtype)
        res[:2] = np.std(self.endog)
        res[2] = 1.0
        return res

    def transform_params(self, unconstrained: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        res = np.empty(3, dtype=unconstrained.dtype)
        res[:2] = np.square(unconstrained[:2])
        res[2] = unconstrained[2]
        return res

    def untransform_params(self, constrained: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        res = np.empty(3, dtype=constrained.dtype)
        res[:2] = np.sqrt(constrained[:2])
        res[2] = constrained[2]
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

        # φ
        self.ssm["transition", 0, 0] = params[2]


# %%
mod = LocalRatioModel(y)
res_sm = mod.fit()

# %%
res_sm.params
