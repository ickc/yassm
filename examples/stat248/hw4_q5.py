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
# # HW 4: Problem 5
#
# *Originally written on 2022-04-09.*

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stat248_data import load_Prob5HW4DataStat248, size

from yassm.linear_gaussian.model.local_ratio import local_ratio_model

# %%
df = load_Prob5HW4DataStat248()

# %%
df

# %%
ax = df.plot()

# %%
y = df.y.values
t = df.index.values

# %% [markdown]
# ## Part d
#
# This is modeled from previous assignments, where we called "local ratio model" here. Reusing that code,

# %%
# %%time
n_sample_used = size(10_000, fast=100)
gibbs_burn_in = size(2_000_000, fast=5_000)
fit = local_ratio_model.fit(
    y,
    None,
    0.0,
    1e1,
    # mle_method="scipy_minimize",
    mle_method="EM",
    mle_n_iter=1000,
    mle_return_chain=True,
    # mle_kwargs={"bounds": ((0.0, None), (0.0, None), (0.0, None))},
    n_sample=gibbs_burn_in + n_sample_used,
    gibbs_burn_in=gibbs_burn_in,
    t=t,
)

# %%
ax = fit.mle_params_chain_to_frame.plot(logx=True, logy=True, title="EM optimization chain")

# %%
fig, ax = plt.subplots()
pd.Series(fit.log_likelihoods(fit.mle_params_chain)).plot(
    title="log-likelihood from EM optimization chain",
    logx=True,
    ax=ax,
)

# %% [markdown]
# For Gibbs sampling, the main function responsible
# are
#
# -   `yassm.lib.linear_gaussian.model.local_ratio._backward_sampling_1d_inplace`
#     for FFBSampling algorithm (step 1 of Gibbs sampling,
# -   `yassm.lib.linear_gaussian.model.local_ratio._sampling_parameters_1d_inplace`
#     for step 2 of Gibbs sampling, and
# -   `yassm.lib.linear_gaussian.model.local_ratio.gibbs_sampling`
#     for the Gibbs sampler.

# %%
# %%time
fit._run_gibbs_sampling(1.6)

# %% [raw]
# %%time
# # keep how many σ of chain
# n_σ = 1.6
# fit._run_gibbs_sampling()
# mask = fit.log_likelihoods(fit.gibbs_params_chain_with_burn_in) > fit.log_likelihood - 0.5 * n_σ**2
# fit.n_sample = mask.sum()
# fit.gibbs_burn_in = fit.n_sample - n_sample_used
# fit.gibbs_params_chain_with_burn_in = np.ascontiguousarray(fit.gibbs_params_chain_with_burn_in[mask])
# fit.gibbs_X_chain_with_burn_in = np.ascontiguousarray(fit.gibbs_X_chain_with_burn_in[mask])
# fit._gibbs_params_chain = fit.gibbs_params_chain_with_burn_in[fit.gibbs_burn_in :]
# fit._gibbs_X_chain = fit.gibbs_X_chain_with_burn_in[fit.gibbs_burn_in :]

# %%
fig, ax = plt.subplots()
pd.Series(fit.log_likelihoods(fit.gibbs_params_chain)).plot(
    title="log-likelihood from Gibbs chain",
    ax=ax,
)

# %%
fig, axes = fit.plot(
    α=0.05,
    alpha=0.5,
    plot_obs=False,
    plot_smoothed_CI_with_obs_cov=False,
    plot_smoothed_from_gibbs_samples=True,
    plot_smoothed_from_gibbs_samples_CI=True,
)

# %% [markdown] tags=[]
# ## Part e

# %%
# σ_ε_sq, σ_Z_sq, φ = params
# taking sqrt of first 2 params
params_chain = fit.gibbs_params_chain
σs = np.append(np.sqrt(params_chain[:, :2]), params_chain[:, 2].reshape(-1, 1), axis=1)

# %%
names = ["σ_ε", "σ_Z", "φ"]

# %%
mle = np.append(np.sqrt(fit.params[:2]), fit.params[2])
df = pd.Series(mle, index=names).to_frame("MLE")

# %%
df["Gibbs-sampling-mean"] = σs.mean(0)
df["Gibbs-sampling-median"] = np.median(σs, 0)
df["Gibbs-sampling-std"] = σs.std(0)

# %%
fig, axes = plt.subplots(len(names))
for i, ax in enumerate(axes):
    sns.histplot(σs[:, i], kde=True, ax=ax)
    ax.set_title(names[i])

# %%
df

# %% [markdown]
# Note that the long tail results in deviations between MLE and mean of $σ_Z, φ$.
