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
# # HW 5: Problem 7
#
# *Originally written on 2022-04-27.*

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit
from stat248_data import load_Prob7HW5DataStat248, size

from yassm.lib.linear_gaussian import prob_normal_1d
from yassm.particle_filter import GeneralNonLinearBootstrapParticleFilterStateSpaceModel

if TYPE_CHECKING:
    import numpy.typing as npt

# %%
df = load_Prob7HW5DataStat248()
y = df.y.values
t = df.index.values
df

# %%
df.plot()

# %% [markdown] tags=[]
# ## Part a
#
# Since $X_0$ is not provided, we will assume $X_0 \sim N(μ_0, Γ_0)$.

# %%
μ_0 = 0.0
σ_0 = 10.0
μ_0, σ_0


# %%
@jit("(f8[::1], f8[::1], f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def pdf_Y_inplace(pdf, y_t, x, params, t):
    # this might overflow
    cov_inv = np.exp(-x.flatten())
    # if cov_inv is infinity, then pdf is a Dirac-delta where for any practical y_t the pdf is 0.
    pdf[:] = np.where(np.isposinf(cov_inv), 0.0, prob_normal_1d(y_t[0], 0.0, cov_inv, False))


@jit("(f8[:, ::1], f8[:, ::1], f8[::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def K_t_inplace(x_t, x_tMinus1, u_t, params, t):
    x_t[:] = x_tMinus1 + u_t.reshape(-1, 1)


@jit("(f8[:, ::1], f8[::1])", nopython=True, nogil=True, cache=True)
def sampling_X0_tilde_inplace(x_0_tilde, params):
    N = x_0_tilde.shape[0]
    x_0_tilde[:, 0] = np.random.normal(μ_0, σ_0, N)


@jit("(f8[::1], i8, i8)", nopython=True, nogil=True, cache=True)
def sampling_U_t(params, t, N):
    α = params[0]

    u_t = np.zeros(N)
    for i in range(N):
        if np.random.uniform(0.0, 1.0) < α:
            u_t[i] = np.random.standard_cauchy()
    return u_t


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
    # this model always predicts the mean to be zero
    return np.zeros((size, N, 1))


# %%
stochastic_volatility_mixture_cauchy_model = GeneralNonLinearBootstrapParticleFilterStateSpaceModel(
    1,
    pdf_Y_inplace=pdf_Y_inplace,
    K_t_inplace=K_t_inplace,
    sampling_X0_tilde_inplace=sampling_X0_tilde_inplace,
    sampling_U_t=sampling_U_t,
    state_to_obs=state_to_obs,
    param_names=["α"],
    particle_filter_method="bootstrap",
)

# %% [markdown] tags=[]
# ### Grid search—1st pass
#
# Grid search in wide-range in log-scale first to get near the MLE.

# %%
n_grid = size(100, fast=4)
n_sample = size(100_000, fast=200)

# %%
αs = np.logspace(-10.0, 0.0, n_grid)

# %%
# %%time
fit = stochastic_volatility_mixture_cauchy_model.fit(
    y,
    [0.5],
    mle_method="grid_search",
    mle_n_iter=1,
    mle_return_chain=True,
    mle_kwargs={"params_grid": αs.reshape(-1, 1)},
    n_sample=n_sample,
    t=t,
)

# %%
log_likelihoods = -fit.neg_log_likelihoods_grid.flatten()

# %%
df_log_likelihoods = pd.Series(log_likelihoods, index=αs).to_frame("Bootstrap particle filter log-likelihood")
df_log_likelihoods.index.name = "α"
df_log_likelihoods.plot(logx=True)

# %%
fit.params_to_series.to_frame("MLE")

# %% [markdown]
# ### Grid search—2nd pass
#
# Search near the previous MLE.

# %%
n_grid = size(100, fast=4)
n_sample = size(1_000_000, fast=200)

# %%
# a width within an order of magnitude
α_min = fit.params[0] / np.sqrt(10.0)
α_max = fit.params[0] * np.sqrt(10.0)
α_min, α_max

# %%
αs = np.linspace(α_min, α_max, n_grid)

# %%
# %%time
fit = stochastic_volatility_mixture_cauchy_model.fit(
    y,
    [0.5],
    mle_method="grid_search",
    mle_n_iter=1,
    mle_return_chain=True,
    mle_kwargs={"params_grid": αs.reshape(-1, 1)},
    n_sample=n_sample,
    t=t,
)

# %%
log_likelihoods = -fit.neg_log_likelihoods_grid.flatten()

# %%
fig, ax = plt.subplots()
df_log_likelihoods = pd.Series(log_likelihoods, index=αs).to_frame("Bootstrap particle filter log-likelihood")
df_log_likelihoods.index.name = "α"
df_log_likelihoods.plot(ax=ax)

# %%
fit.params_to_series.to_frame("MLE")

# %% [markdown]
# ## Part b
#
# We can try to visualize how good the fitting is by comparing to $log(Y^2)$, because
#
# $$Y^2 | X=x ~\sim e^x χ^2(1),$$
#
# $$\mathbb{E}[Y^2 | X=x] = e^x,$$
#
# and $X_t$ is piece-wise constant.

# %%
fig, ax = plt.subplots()
pd.DataFrame(
    {
        "log(y^2)": np.log(np.square(y)),
        "filtering mean": fit.x_t.mean(axis=1).flatten(),
    },
).plot(ax=ax)
