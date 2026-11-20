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
# # HW 5: Problem 1
#
# *Originally written on 2022-04-21.*

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit
from stat248_data import load_nile, size

from yassm.linear_gaussian.model.local_level import local_level_model
from yassm.particle_filter import GeneralLinearGaussianParticleFilterStateSpaceModel

if TYPE_CHECKING:
    import numpy.typing as npt

# %%
df = load_nile()
y = df.Nile.values
t = df.index.values
df

# %%
df.plot()

# %% [markdown]
# ## Part a
#
# Reusing the code from previous assignments.

# %%
n_sample = size(1_000_000, fast=200)

y_max = y.max()
y_min = y.min()
y_mid = 0.5 * (y_max + y_min)
y_rge = 0.5 * (y_max - y_min)
μ_0 = y_mid
Γ_0 = y_rge * 10.0

μ_0, Γ_0

# %%
fit = local_level_model.fit(y, None, μ_0, Γ_0, mle_n_iter=1000, mle_return_chain=True, t=t)
fit.mle_params_chain_to_frame.plot(logx=True, logy=True)

# %%
ans = np.sqrt(fit.params_to_series.to_frame("MLE"))
ans.index = ["σ_ε", "σ_Z"]
ans


# %% [markdown]
# ## Part b
#
# Particle filters with general linear Gaussian assumptions are implemented at
# `yassm.particle_filter.GeneralLinearGaussianParticleFilterStateSpaceModel`.
#
# Then we specialize this to the local level model by defining these below,


# %%
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def A0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.ones((1, 1))


@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def B0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.ones((1, 1))


# state cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def Σ0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    res = np.empty((1, 1))
    res[0, 0] = params[1]
    return res


# obs. cov.
@jit("f8[:, ::1](f8[::1])", nopython=True, nogil=True, cache=True)
def R0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    res = np.empty((1, 1))
    res[0, 0] = params[0]
    return res


@jit("Tuple((f8[::1], f8[:, ::1]))(f8[::1])", nopython=True, nogil=True, cache=True)
def X0(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    temp = np.empty((2, 1))
    μ = temp[0]
    Γ = temp[1].reshape((1, 1))
    μ[0] = μ_0
    Γ[0, 0] = Γ_0
    return μ, Γ


# %%
bootstrap_local_level_model = GeneralLinearGaussianParticleFilterStateSpaceModel(
    2,
    A0,
    B0,
    Σ0,
    R0,
    X0=X0,
    param_names=["obs_cov", "state_cov"],
    particle_filter_method="bootstrap",
)
bootstrap_local_level_model.params_guess = local_level_model.params_guess

# %%
fit_bootstrap = bootstrap_local_level_model.fit(
    y,
    fit.params,
    n_sample=n_sample,
    t=t,
    param_names=local_level_model.param_names,
)

# %%
# %%time
fit_bootstrap._run_particle_filter()

# %%
fig, axes = plt.subplots(2, figsize=(6, 9))
ax = axes[0]
fit_bootstrap.plot(
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
)
ax.set_title("Bootstrap particle filter")
ax = axes[1]
fit.plot(
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
)
ax.set_title("Kalman filter")

# %%
pd.DataFrame(
    {
        "Kalman filter": fit.m_t_given_t.flatten(),
        "Bootstrap particle filter": fit_bootstrap.m_t_given_t.flatten(),
    }
).plot(title="Filtering mean")

# %%
pd.DataFrame(
    {
        "Kalman filter": np.sqrt(fit.Q_t_given_t.flatten()),
        "Bootstrap particle filter": np.sqrt(fit_bootstrap.Q_t_given_t.flatten()),
    }
).plot(title="Filtering std")

# %% [markdown]
# ## Part c

# %%
optimal_local_level_model = GeneralLinearGaussianParticleFilterStateSpaceModel(
    2,
    A0,
    B0,
    Σ0,
    R0,
    X0=X0,
    param_names=["obs_cov", "state_cov"],
    particle_filter_method="optimal",
)
optimal_local_level_model.params_guess = local_level_model.params_guess

# %%
fit_optimal = optimal_local_level_model.fit(
    y,
    fit.params,
    n_sample=n_sample,
    t=t,
    param_names=local_level_model.param_names,
)

# %%
# %%time
fit_optimal._run_particle_filter()

# %%
fig, axes = plt.subplots(3, figsize=(9, 9))
ax = axes[0]
fit_bootstrap.plot(
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
)
ax.set_title("Bootstrap particle filter")
ax = axes[1]
fit.plot(
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
)
ax.set_title("Kalman filter")
ax = axes[2]
fit_optimal.plot(
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
)
ax.set_title("Optimal particle filter")

# %%
pd.DataFrame(
    {
        "Kalman filter": fit.m_t_given_t.flatten(),
        "Bootstrap particle filter": fit_bootstrap.m_t_given_t.flatten(),
        "Optimal particle filter": fit_optimal.m_t_given_t.flatten(),
    }
).plot(title="Filtering mean")

# %%
pd.DataFrame(
    {
        "Kalman filter": np.sqrt(fit.Q_t_given_t.flatten()),
        "Bootstrap particle filter": np.sqrt(fit_bootstrap.Q_t_given_t.flatten()),
        "Optimal particle filter": np.sqrt(fit_optimal.Q_t_given_t.flatten()),
    }
).plot(title="Filtering std")

# %% [markdown]
# ## Part d
#
# We will just do a grid search around the MLE from linear Gaussian MLE,
# because the MLE from particle filter are too noisy for automatic minimization/maximization.

# %%
n_grid = size(100, fast=4)
n_sample = size(10_000, fast=200)

# %%
np.sqrt(fit.params)

# %%
σ_εs, σ_Ζs = np.meshgrid(
    np.linspace(100.0, 200.0, n_grid),
    np.linspace(25.0, 50.0, n_grid),
)
# σ_sq
params = np.square(np.column_stack((σ_εs.flatten(), σ_Ζs.flatten())))

# %%
# %%time
fit_bootstrap_mle = bootstrap_local_level_model.fit(
    y,
    None,
    mle_method="grid_search",
    mle_n_iter=1,
    mle_return_chain=True,
    mle_kwargs={"params_grid": params},
    n_sample=n_sample,
    t=t,
    param_names=local_level_model.param_names,
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

# %% [markdown] tags=[]
# ## Part e

# %%
# %%time
fit_optimal_mle = optimal_local_level_model.fit(
    y,
    None,
    mle_method="grid_search",
    mle_n_iter=1,
    mle_return_chain=True,
    mle_kwargs={"params_grid": params},
    n_sample=n_sample,
    t=t,
    param_names=local_level_model.param_names,
)

# %%
log_likelihoods_2d_optimal = -fit_optimal_mle.neg_log_likelihoods_grid.reshape((n_grid, n_grid))

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(σ_εs, σ_Ζs, log_likelihoods_2d_optimal)
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
