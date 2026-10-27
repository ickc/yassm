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
# # HW 5: Problem 3
#
# *Originally written on 2022-04-24.*

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit
from stat248_data import load_googletrends21Jan2016, size

from yassm.lib.linear_gaussian.model.local_linear import A0, B0, R0, Σ0
from yassm.linear_gaussian.model.local_linear import local_linear_model
from yassm.particle_filter import GeneralLinearGaussianParticleFilterStateSpaceModel

if TYPE_CHECKING:
    import numpy.typing as npt

# %%
df = load_googletrends21Jan2016()
df_train = df.iloc[:529]
df_test = df.iloc[529:]
y = df_train.google.values
t = df_train.index.values
df

# %%
df_train.plot()

# %% [markdown]
# ## Part a

# %%
y_max = y.max()
y_min = y.min()
y_mid = 0.5 * (y_max + y_min)
y_rge = 0.5 * (y_max - y_min)
μ_0 = np.full(2, y_mid)
Γ_0 = (y_rge * y_rge) * np.identity(2)

μ_0, Γ_0

# %%
fit = local_linear_model.fit(y, None, μ_0, Γ_0, mle_n_iter=1, mle_return_chain=True, t=t)
fit.mle_params_chain_to_frame.plot(logx=True, logy=True)

# %%
ans = np.sqrt(fit.params_to_series.to_frame("MLE"))
ans.index = ["σ_ε", "σ_Z"]
ans

# %% [markdown]
# ## Part b
#
# Similar to problem 1,
# we will just do a grid search around the MLE from linear Gaussian MLE,
# because the MLE from particle filter are too noisy for automatic minimization/maximization.
#
# We use more samples but less refined grid for bootstrap as it is more noisy.

# %%
n_grid = size(10, fast=4)
n_sample = size(100_000, fast=200)


# %%
@jit(nopython=True, nogil=True, cache=True)
def X0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return μ_0, Γ_0


# %%
bootstrap_local_linear_model = GeneralLinearGaussianParticleFilterStateSpaceModel(
    2,
    A0,
    B0,
    Σ0,
    R0,
    X0=X0,
    param_names=local_linear_model.param_names,
    particle_filter_method="bootstrap",
)
bootstrap_local_linear_model.params_guess = local_linear_model.params_guess

# %%
np.sqrt(fit.params)

# %%
σ_εs, σ_Ζs = np.meshgrid(
    np.linspace(0.5, 1.5, n_grid),
    np.linspace(0.10, 0.30, n_grid),
)
# σ_sq
params = np.square(np.column_stack((σ_εs.flatten(), σ_Ζs.flatten())))

# %%
# %%time
fit_bootstrap_mle = bootstrap_local_linear_model.fit(
    y,
    None,
    mle_method="grid_search",
    mle_n_iter=1,
    mle_return_chain=True,
    mle_kwargs={"params_grid": params},
    n_sample=n_sample,
    t=t,
)

# %%
log_likelihoods_2d = -fit_bootstrap_mle.neg_log_likelihoods_grid.reshape((n_grid, n_grid))

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(σ_εs, σ_Ζs, log_likelihoods_2d)
ax.set_title("Bootstrap particle filter log-likelihood")
ax.set_xlabel("σ_ε")
ax.set_ylabel("σ_Ζ")

# %%
np.sqrt(
    pd.DataFrame(
        {
            "Kalman filter": fit.params,
            "Bootstrap particle filter": fit_bootstrap_mle.params,
        },
        index=ans.index,
    )
)

# %% [markdown]
# ## Part c

# %%
fit_bootstrap = bootstrap_local_linear_model.fit(
    y,
    fit.params,
    n_sample=n_sample,
    t=t,
)

# %%
# %%time
fit_bootstrap._run_particle_filter()

# %%
fig, axes = plt.subplots(2, figsize=(6, 9))
ax = axes[0]
fit_bootstrap.plot(
    T_forecast=100,
    plot_obs=False,
    plot_smoothed_CI_with_obs_cov=False,
    plot_filtered=True,
    plot_filtered_CI=True,
    plot_smoothed=False,
    plot_smoothed_CI=False,
    plot_smoothed_from_samples=False,
    plot_smoothed_from_samples_CI=False,
    plot_smoothed_from_gibbs_samples=False,
    plot_smoothed_from_gibbs_samples_CI=False,
    n_std=1.0,
    axes=[ax],
    forecast_CI_with_obs_cov=False,
)
ax.set_title("Bootstrap particle filter")
ax = axes[1]
fit.plot(
    T_forecast=100,
    plot_obs=False,
    plot_smoothed_CI_with_obs_cov=False,
    plot_filtered=True,
    plot_filtered_CI=True,
    plot_smoothed=False,
    plot_smoothed_CI=False,
    plot_smoothed_from_samples=False,
    plot_smoothed_from_samples_CI=False,
    plot_smoothed_from_gibbs_samples=False,
    plot_smoothed_from_gibbs_samples_CI=False,
    n_std=1.0,
    axes=[ax],
    forecast_CI_with_obs_cov=False,
)
ax.set_title("Kalman filter")

# %% [markdown]
# ## Part d
#
# We use lesser samples but more refined grid for optimal as it is less noisy.

# %%
n_grid = size(100, fast=4)
n_sample = size(1_000, fast=200)

# %%
optimal_local_linear_model = GeneralLinearGaussianParticleFilterStateSpaceModel(
    2,
    A0,
    B0,
    Σ0,
    R0,
    X0=X0,
    param_names=local_linear_model.param_names,
    particle_filter_method="optimal",
)
optimal_local_linear_model.params_guess = local_linear_model.params_guess

# %%
σ_εs, σ_Ζs = np.meshgrid(
    np.linspace(0.5, 1.5, n_grid),
    np.linspace(0.10, 0.30, n_grid),
)
# σ_sq
params = np.square(np.column_stack((σ_εs.flatten(), σ_Ζs.flatten())))

# %%
# %%time
fit_optimal_mle = optimal_local_linear_model.fit(
    y,
    None,
    mle_method="grid_search",
    mle_n_iter=1,
    mle_return_chain=True,
    mle_kwargs={"params_grid": params},
    n_sample=n_sample,
    t=t,
)

# %%
log_likelihoods_2d = -fit_optimal_mle.neg_log_likelihoods_grid.reshape((n_grid, n_grid))

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(σ_εs, σ_Ζs, log_likelihoods_2d)
ax.set_title("Optimal particle filter log-likelihood")
ax.set_xlabel("σ_ε")
ax.set_ylabel("σ_Ζ")

# %%
np.sqrt(
    pd.DataFrame(
        {
            "Kalman filter": fit.params,
            "Bootstrap particle filter": fit_bootstrap_mle.params,
            "Optimal particle filter": fit_optimal_mle.params,
        },
        index=ans.index,
    )
)

# %% [markdown]
# ## Part e

# %%
n_sample = size(10_000, fast=200)

# %%
# %%time
fit_optimal = optimal_local_linear_model.fit(
    y,
    fit.params,
    n_sample=n_sample,
    t=t,
)

# %%
# %%time
fit_optimal._run_particle_filter()

# %%
fig, axes = plt.subplots(3, figsize=(9, 9))
ax = axes[0]
fit_bootstrap.plot(
    T_forecast=100,
    plot_obs=False,
    plot_smoothed_CI_with_obs_cov=False,
    plot_filtered=True,
    plot_filtered_CI=True,
    plot_smoothed=False,
    plot_smoothed_CI=False,
    plot_smoothed_from_samples=False,
    plot_smoothed_from_samples_CI=False,
    plot_smoothed_from_gibbs_samples=False,
    plot_smoothed_from_gibbs_samples_CI=False,
    n_std=1.0,
    axes=[ax],
    forecast_CI_with_obs_cov=False,
)
ax.set_title("Bootstrap particle filter")
ax = axes[1]
fit.plot(
    T_forecast=100,
    plot_obs=False,
    plot_smoothed_CI_with_obs_cov=False,
    plot_filtered=True,
    plot_filtered_CI=True,
    plot_smoothed=False,
    plot_smoothed_CI=False,
    plot_smoothed_from_samples=False,
    plot_smoothed_from_samples_CI=False,
    plot_smoothed_from_gibbs_samples=False,
    plot_smoothed_from_gibbs_samples_CI=False,
    n_std=1.0,
    axes=[ax],
    forecast_CI_with_obs_cov=False,
)
ax.set_title("Kalman filter")
ax = axes[2]
fit_optimal.plot(
    T_forecast=100,
    plot_obs=False,
    plot_smoothed_CI_with_obs_cov=False,
    plot_filtered=True,
    plot_filtered_CI=True,
    plot_smoothed=False,
    plot_smoothed_CI=False,
    plot_smoothed_from_samples=False,
    plot_smoothed_from_samples_CI=False,
    plot_smoothed_from_gibbs_samples=False,
    plot_smoothed_from_gibbs_samples_CI=False,
    n_std=1.0,
    axes=[ax],
    forecast_CI_with_obs_cov=False,
)
ax.set_title("Optimal particle filter")

# %% [markdown]
# ## Part f
#
# The blue curve is the actual data.

# %%
fig, axes = plt.subplots(3, figsize=(9, 9))
ax = axes[0]
fit_bootstrap.plot(
    T_forecast=100,
    plot_obs=False,
    plot_smoothed_CI_with_obs_cov=False,
    plot_filtered=True,
    plot_filtered_CI=True,
    plot_smoothed=False,
    plot_smoothed_CI=False,
    plot_smoothed_from_samples=False,
    plot_smoothed_from_samples_CI=False,
    plot_smoothed_from_gibbs_samples=False,
    plot_smoothed_from_gibbs_samples_CI=False,
    n_std=1.0,
    axes=[ax],
    forecast_CI_with_obs_cov=False,
)
ax.set_title("Bootstrap particle filter")
df.plot(ax=ax)
ax = axes[1]
fit.plot(
    T_forecast=100,
    plot_obs=False,
    plot_smoothed_CI_with_obs_cov=False,
    plot_filtered=True,
    plot_filtered_CI=True,
    plot_smoothed=False,
    plot_smoothed_CI=False,
    plot_smoothed_from_samples=False,
    plot_smoothed_from_samples_CI=False,
    plot_smoothed_from_gibbs_samples=False,
    plot_smoothed_from_gibbs_samples_CI=False,
    n_std=1.0,
    axes=[ax],
    forecast_CI_with_obs_cov=False,
)
ax.set_title("Kalman filter")
df.plot(ax=ax)
ax = axes[2]
fit_optimal.plot(
    T_forecast=100,
    plot_obs=False,
    plot_smoothed_CI_with_obs_cov=False,
    plot_filtered=True,
    plot_filtered_CI=True,
    plot_smoothed=False,
    plot_smoothed_CI=False,
    plot_smoothed_from_samples=False,
    plot_smoothed_from_samples_CI=False,
    plot_smoothed_from_gibbs_samples=False,
    plot_smoothed_from_gibbs_samples_CI=False,
    n_std=1.0,
    axes=[ax],
    forecast_CI_with_obs_cov=False,
)
ax.set_title("Optimal particle filter")
df.plot(ax=ax)
