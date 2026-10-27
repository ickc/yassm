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
# # HW 4: Problem 2
#
# *Originally written on 2022-04-07.*

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from stat248_data import load_nile

# %%
from yassm.linear_gaussian.model.local_level import local_level_model

# %%
df = load_nile()
df

# %%
df.plot()

# %%
y = df.Nile.values.astype(np.float64)
t = df.index.values

# %% [markdown]
# ## Part a
#
# Main function implementing the Gibbs sampler algorithm is in
#
# -   `yassm.lib.linear_gaussian.model.local_level.gibbs_sampling`.
#
# See documentation attached at the end.
#
# The results of this sampling is in `fit.gibbs_params_chain, fit.gibbs_X_chain`.

# %%
# %%time
n_sample = 100000
n = 500
fit = local_level_model.fit(y, None, 0.0, np.inf, mle_n_iter=1000, n_sample=n_sample, gibbs_burn_in=n_sample - n)
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
# Main function implementing the CI is
#
# -   `yassm.linear_gaussian.GeneralLinearGaussianStateSpaceModelFit.smoothed_obs_CI_from_gibbs_samples`,
#
# called by the plot function below.
#
# See documentation attached at the end for details.

# %%
fit.plot(
    α=0.05,
    alpha=0.5,
    plot_smoothed_CI_with_obs_cov=False,
    plot_smoothed_from_gibbs_samples=True,
    plot_smoothed_from_gibbs_samples_CI=True,
)
