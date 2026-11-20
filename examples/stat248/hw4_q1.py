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
# # HW 4: Problem 1
#
# *Originally written on 2022-04-10.*

# %%
from __future__ import annotations

import numpy as np
import pandas as pd
from stat248_data import load_googletrends21Jan2016

# %%
from yassm.linear_gaussian.model.local_linear import local_linear_model

# %%
df = load_googletrends21Jan2016()
df

# %%
ax = df.google.plot(
    xlabel="Weekly Time",
    ylabel="Google Search Popularity",
    title="Google Trends Data for the query google",
)

# %%
y = df.google.values
t = df.index.values

# %% [markdown]
# ## Part a
#
# Reusing the tool we developed for previous assignments,

# %%
C = 1e8
μ_0 = np.zeros(2)
Γ_0 = C * np.identity(2)

# %%
fit = local_linear_model.fit(
    y,
    None,
    μ_0,
    Γ_0,
    mle_n_iter=1,
    n_sample=200,
    mle_return_chain=True,
    t=t,
)

# %%
fit.mle_params_chain_to_frame.plot(logx=True, logy=True)

# %%
fit.plot()

# %%
obs_std, state_std = np.sqrt(fit.params)


pd.Series({"σ_ε": obs_std, "σ_Z": state_std})

# %% [markdown]
# ## Part b
#
# Main function implementing FFBSampling algorithm is in
#
# -   `yassm.linear_gaussian.GeneralLinearGaussianStateSpaceModel.backward_sampling`.
#
# See documentation attached at the end.

# %% [markdown]
# Plotting as a function of time,

# %%
pd.DataFrame(
    {
        "sample": fit.smoothed_obs_from_samples.flatten(),
        "actual": fit.smoothed_obs.flatten(),
    }
).plot(title="Sample mean vs Kalman Smoother mean")

# %%
pd.DataFrame(
    {
        "sample": fit.smoothed_obs_cov_from_samples.flatten(),
        "actual": fit.smoothed_obs_cov.flatten(),
    }
).plot(title="Sample covariance vs Kalman Smoother covariance")

# %% [markdown]
# Plotting as scatter plot,

# %%
pd.Series(
    fit.smoothed_obs_from_samples.flatten(),
    index=fit.smoothed_obs.flatten(),
).plot(title="Sample mean vs Kalman Smoother mean")

# %%
pd.Series(
    fit.smoothed_obs_cov_from_samples.flatten(),
    index=fit.smoothed_obs_cov.flatten(),
).plot(title="Sample covariance vs Kalman Smoother covariance", style="o", logx=True, logy=True)
