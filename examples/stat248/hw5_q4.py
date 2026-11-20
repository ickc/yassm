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
# # HW 5: Problem 4
#
# *Originally written on 2022-04-25.*

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from numba import jit
from stat248_data import load_googletrends21Jan2016, size

from yassm.lib.linear_gaussian import prob_cauchy
from yassm.linear_gaussian.model.local_linear import local_linear_model
from yassm.particle_filter import GeneralNonLinearBootstrapParticleFilterStateSpaceModel

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
C_sqrt = y.max()
μ_0 = np.zeros(2)
Γ_0 = (C_sqrt * C_sqrt) * np.identity(2)
μ_0, Γ_0, C_sqrt

# %%
A = np.array(
    [
        [2.0, -1.0],
        [1.0, 0.0],
    ],
)
A_T = np.ascontiguousarray(A.T)


@jit("(f8[::1], f8[::1], f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def pdf_Y_inplace(pdf, y_t, x, params, t):
    τ_ε_sq, τ_Z_sq = params
    pdf[:] = prob_cauchy(y_t[0], np.ascontiguousarray(x[:, 0]), τ_ε_sq, False)


@jit("(f8[:, ::1], f8[:, ::1], f8[::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def K_t_inplace(x_t, x_tMinus1, u_t, params, t):
    x_t[:] = x_tMinus1 @ A_T
    x_t[:, 0] += u_t


@jit("(f8[:, ::1], f8[::1])", nopython=True, nogil=True, cache=True)
def sampling_X0_tilde_inplace(x_0_tilde, params):
    N = x_0_tilde.shape[0]
    x_0_tilde[:] = np.random.normal(0.0, C_sqrt, (N, 2))


@jit("f8[::1](f8[::1], i8, i8)", nopython=True, nogil=True, cache=True)
def sampling_U_t(params, t, N):
    τ_ε_sq, τ_Z_sq = params
    return np.sqrt(τ_Z_sq) * np.random.standard_cauchy(N)


@jit(
    "f8[:, :, ::1](f8[::1], f8[:, :, ::1], int64)",
    nopython=True,
    nogil=True,
    cache=True,
)
def state_to_obs(
    params: npt.NDArray[np.float64],
    x_t: npt.NDArray[np.float64],
    t0: int,
) -> npt.NDArray[np.float64]:
    size, N, d = x_t.shape
    return np.ascontiguousarray(x_t[:, :, 0]).reshape(size, N, 1)


# %%
@jit("f8[::1](f8[::1])", nopython=True, nogil=True, cache=True)
def transform_params(unconstrained: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.exp(unconstrained)


@jit("f8[::1](f8[::1])", nopython=True, nogil=True, cache=True)
def untransform_params(constrainted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.log(constrainted)


# %%
local_linear_cauchy_model = GeneralNonLinearBootstrapParticleFilterStateSpaceModel(
    2,
    d=2,
    pdf_Y_inplace=pdf_Y_inplace,
    K_t_inplace=K_t_inplace,
    sampling_X0_tilde_inplace=sampling_X0_tilde_inplace,
    sampling_U_t=sampling_U_t,
    state_to_obs=state_to_obs,
    param_names=["τ_ε_sq", "τ_Z_sq"],
    particle_filter_method="bootstrap",
    transform_params=transform_params,
    untransform_params=untransform_params,
)

# %% [markdown]
# ### Automatic MLE
#
# This takes the numerical negative log-likihood and minimize using `scipy.optimize.minimize`.
#
# From the experience of problem 2, we'll only try Powell and COBYLA.

# %%
n_sample = size(100_000, fast=200)

# %%
# %%time
res = {}
for method in (
    # "Nelder-Mead",
    "Powell",
    # "CG",
    # "BFGS",
    # "Newton-CG",
    # "L-BFGS-B",
    # "TNC",
    "COBYLA",
    # "SLSQP",
    # "trust-const",
    # "dogleg",
    # "trust-ncg",
    # "trust-exact",
    # "trust-krylov",
):
    try:
        fit = local_linear_cauchy_model.fit(
            y,
            [y.std(), y.std()],
            mle_method="scipy_minimize",
            mle_n_iter=1,
            mle_return_chain=True,
            mle_kwargs={"method": method},
            n_sample=n_sample,
            t=t,
        )
        print(f"method {method} resulted in MLE: {fit.params}.")
        display(fit.mle_params_chain_to_frame)
        res[method] = fit
    except Exception:
        pass

# %%
method = "COBYLA"
fit_optimized = res[method]

# %%
fit_optimized.mle_params_chain_to_frame.plot(logx=True, logy=True)

# %%
ans = np.sqrt(fit_optimized.params_to_series.to_frame(method))
ans.index = ["τ_ε", "τ_Z"]
ans

# %% [markdown]
# ### Grid search
#
# Then we use the grid to verify this is indeed a local MLE and obtain a more refined MLE.

# %%
n_grid = size(100, fast=4)
n_sample = size(100_000, fast=200)

# %%
τ_εs, τ_Ζs = np.meshgrid(
    np.linspace(0.1, 0.5, n_grid),
    np.linspace(0.01, 0.05, n_grid),
)
# τ_sq
params = np.square(np.column_stack((τ_εs.flatten(), τ_Ζs.flatten())))

# %%
# %%time
fit = local_linear_cauchy_model.fit(
    y,
    [y.std(), y.std()],
    mle_method="grid_search",
    mle_n_iter=1,
    mle_return_chain=True,
    mle_kwargs={"params_grid": params},
    n_sample=n_sample,
    t=t,
)

# %%
log_likelihoods_2d = -fit.neg_log_likelihoods_grid.reshape((n_grid, n_grid))

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(τ_εs, τ_Ζs, log_likelihoods_2d)
ax.set_title("Bootstrap particle filter log-likelihood")
ax.set_xlabel("τ_ε")
ax.set_ylabel("τ_Ζ")

# %%
ans_sq = pd.DataFrame(
    [fit_optimized.params, fit.params], columns=["τ_ε_sq", "τ_Z_sq"], index=["scipy_optimize", "grid_search"]
)
ans_sq

# %%
ans = np.sqrt(ans_sq)
ans.columns = ["τ_ε", "τ_Z"]
ans

# %% [markdown]
# After automatic minimization lands us near the MLE, the grid-search is able to refine the value.
# We'll use the grid-search MLE below.

# %% [markdown]
# ## Part b

# %%
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
    forecast_CI_with_obs_cov=False,
)

# %% [markdown]
# ## Part c

# %%
fit_kalman = local_linear_model.fit(
    y,
    None,
    μ_0=μ_0,
    Γ_0=Γ_0,
    mle_n_iter=1,
    mle_return_chain=False,
    t=t,
)

# %%
fig, axes = plt.subplots(2, figsize=(6, 9))
ax = axes[0]
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
    plot_forecast_CI=False,
    n_std=1.0,
    axes=[ax],
    forecast_CI_with_obs_cov=False,
)
ax.set_title("Bootstrap particle filter")
df.plot(ax=ax)
ax = axes[1]
fit_kalman.plot(
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
