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
# # HW 4: Problem 3
#
# *Originally written on 2022-04-08.*

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numba import jit
from stat248_data import load_Prob3HW4DataStat248, size

from yassm.linear_gaussian.model.local_linear import local_linear_model

# %%
df_ts = load_Prob3HW4DataStat248()

# %%
n = df_ts.shape[0]
y = df_ts.y.values
t = df_ts.index.values.astype(np.uint16)
np.testing.assert_array_equal(t, np.arange(1, n + 1))
u = (t - 1) / n


# %%
@jit("f8[::1](f8[::1])", nopython=True, nogil=True)
def f(u):
    return np.sin(15.0 * u) + np.exp(-0.5 * np.square(u)) + 0.5 * np.square(u - 0.5)


# %%
df_ts["true-y"] = f(u)

# %%
df_ts

# %%
df_ts.plot()

# %% [markdown]
# See written derivation of the step 2 of Gibbs sampler.
#
# Main functions to sample parameters are
#
# -   `yassm.lib.linear_gaussian.model.local_linear`,
# -   `yassm.linear_gaussian.GeneralLinearGaussianStateSpaceModel.gibbs_sampling`,
#
# where the latter performing Gibbs sampling.
#
# See documentation attached at the end for details.

# %%
n_sample = size(400, fast=50)
n = 800
C = 1e8
μ_0 = np.zeros(2)
Γ_0 = C * np.identity(2)

# %%
# %%time
fit = local_linear_model.fit(
    y,
    None,
    μ_0,
    Γ_0,
    mle_n_iter=1,
    mle_return_chain=True,
    n_sample=n,
    gibbs_burn_in=n - n_sample,
    t=u,
)

# %%
# %%time
params_chain = fit.gibbs_params_chain

# %% [markdown]
# ## Part b

# %%
σs = np.sqrt(params_chain)

# %%
names = ["σ_ε", "σ_Z"]

# %%
fig, axes = plt.subplots(2)
for i, ax in enumerate(axes):
    sns.histplot(σs[:, i], kde=True, ax=ax)
    ax.set_title(names[i])

# %%
df = fit.params_to_series.to_frame("MLE").apply(np.sqrt)
df.index = names

# %%
df["Gibbs-sampling-mean"] = σs.mean(0)
df["Gibbs-sampling-median"] = np.median(σs, 0)

# %%
df

# %% [markdown]
# ## Part c
#
# Reusing from Q2,

# %%
fig, axes = fit.plot(
    α=0.05,
    alpha=0.5,
    plot_obs=False,
    plot_smoothed_CI_with_obs_cov=False,
    plot_smoothed_from_gibbs_samples=True,
    plot_smoothed_from_gibbs_samples_CI=True,
)

# %% [markdown]
# ## Part d

# %%
fig, axes = fit.plot(
    α=0.05,
    alpha=0.5,
    plot_obs=False,
    plot_smoothed=False,
    plot_smoothed_CI=False,
    plot_smoothed_CI_with_obs_cov=False,
    plot_smoothed_from_gibbs_samples=True,
    plot_smoothed_from_gibbs_samples_CI=True,
)
ax = axes[0]
ax.plot(u, df_ts["true-y"], label="true function")
ax.legend()
