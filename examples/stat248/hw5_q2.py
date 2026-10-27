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
# # HW 5: Problem 2
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
from stat248_data import load_nile, size

from yassm.lib.linear_gaussian import prob_cauchy
from yassm.linear_gaussian.model.local_level import local_level_model
from yassm.particle_filter import GeneralNonLinearBootstrapParticleFilterStateSpaceModel

if TYPE_CHECKING:
    import numpy.typing as npt

# %%
df = load_nile()
y = df.Nile.values.reshape(-1, 1).astype(np.float64)
t = df.index.values
df

# %%
df.plot()

# %% [markdown]
# ## Part a

# %%
C_sqrt = y.max()
C_sqrt


# %%
@jit("(f8[::1], f8[::1], f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def pdf_Y_inplace(pdf, y_t, x, params, t):
    τ_ε_sq, τ_Z_sq = params
    pdf[:] = prob_cauchy(y_t[0], x.flatten(), τ_ε_sq, False)


@jit("(f8[:, ::1], f8[:, ::1], f8[::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def K_t_inplace(x_t, x_tMinus1, u_t, params, t):
    x_t[:] = x_tMinus1 + u_t.reshape(-1, 1)


@jit("(f8[:, ::1], f8[::1])", nopython=True, nogil=True, cache=True)
def sampling_X0_tilde_inplace(x_0_tilde, params):
    N = x_0_tilde.shape[0]
    x_0_tilde[:, 0] = np.random.normal(0.0, C_sqrt, N)


@jit("(f8[::1], i8, i8)", nopython=True, nogil=True, cache=True)
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
    return x_t


# %%
@jit("f8[::1](f8[::1])", nopython=True, nogil=True, cache=True)
def transform_params(unconstrained: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.exp(unconstrained)


@jit("f8[::1](f8[::1])", nopython=True, nogil=True, cache=True)
def untransform_params(constrainted: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.log(constrainted)


# %%
local_level_cauchy_model = GeneralNonLinearBootstrapParticleFilterStateSpaceModel(
    2,
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
# We'll use all available methods to attempt automatical optimzation.
#
# This takes the numerical negative log-likihood and minimize using `scipy.optimize.minimize`.

# %%
n_sample = size(10_000, fast=200)

# %%
# %%time
for method in (
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "SLSQP",
    "trust-const",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
):
    try:
        fit = local_level_cauchy_model.fit(
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
    except Exception:
        pass

# %% [markdown]
# Looks like the COBYLA method is the best, and the Powell method is ok too.
#
# From [COBYLA - HandWiki](https://handwiki.org/wiki/COBYLA),
#
# > It works by iteratively approximating the actual constrained optimization problem with linear programming problems. During an iteration, an approximating linear programming problem is solved to obtain a candidate for the optimal solution. The candidate solution is evaluated using the original objective and constraint functions, yielding a new data point in the optimization space. This information is used to improve the approximating linear programming problem used for the next iteration of the algorithm. When the solution cannot be improved anymore, the step size is reduced, refining the search. When the step size becomes sufficiently small, the algorithm finishes.
#
# From [scipy.optimize.minimize — SciPy v1.8.0 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html),
#
# > Method Powell is a modification of Powell’s method which is a conjugate direction method. It performs sequential one-dimensional minimizations along each vector of the directions set (direc field in options and info), which is updated at each iteration of the main minimization loop. The function need not be differentiable, and no derivatives are taken. If bounds are not provided, then an unbounded line search will be used. If bounds are provided and the initial guess is within the bounds, then every function evaluation throughout the minimization procedure will be within the bounds. If bounds are provided, the initial guess is outside the bounds, and direc is full rank (default has full rank), then some function evaluations during the first iteration may be outside the bounds, but every function evaluation after the first iteration will be within the bounds. If direc is not full rank, then some parameters may not be optimized and the solution is not guaranteed to be within the bounds.

# %%
method = "COBYLA"
fit_optimized = local_level_cauchy_model.fit(
    y,
    [y.std(), y.std()],
    mle_method="scipy_minimize",
    mle_n_iter=1,
    mle_return_chain=True,
    mle_kwargs={"method": method},
    n_sample=n_sample,
    t=t,
)

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
    np.linspace(40.0, 100.0, n_grid),
    np.linspace(1.0, 20.0, n_grid),
)
# τ_sq
params = np.square(np.column_stack((τ_εs.flatten(), τ_Ζs.flatten())))

# %%
# %%time
fit = local_level_cauchy_model.fit(
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
fit._run_particle_filter()

# %%
fit.plot(
    plot_smoothed_CI_with_obs_cov=False,
    plot_smoothed=False,
    plot_smoothed_CI=False,
    plot_filtered=True,
    plot_filtered_CI=True,
    n_std=1.0,
)

# %% [markdown]
# ## Part c

# %%
fit_guassian = local_level_model.fit(y.flatten(), None, 0.0, C_sqrt * C_sqrt, mle_n_iter=1, t=t)

# %%
fit_guassian.params_to_series.to_frame("MLE")

# %%
fig, axes = plt.subplots(2, figsize=(6, 9))
ax = axes[0]
fit.plot(
    plot_smoothed_CI_with_obs_cov=False,
    plot_smoothed=False,
    plot_smoothed_CI=False,
    plot_filtered=True,
    plot_filtered_CI=True,
    n_std=1.0,
    axes=[ax],
)
ax.set_title("Bootstrap particle filter w/ Cauchy dist.")
ax = axes[1]
fit_guassian.plot(
    plot_smoothed_CI_with_obs_cov=False,
    plot_smoothed=False,
    plot_smoothed_CI=False,
    plot_filtered=True,
    plot_filtered_CI=True,
    n_std=1.0,
    axes=[ax],
)
ax.set_title("Kalman filter")

# %%
pd.DataFrame(
    {
        "Kalman filter": fit_guassian.m_t_given_t.flatten(),
        "Bootstrap particle filter w/ Cauchy dist.": fit.m_t_given_t.flatten(),
    }
).plot(title="Filtering mean")

# %%
pd.DataFrame(
    {
        "Kalman filter": fit_guassian.Q_t_given_t.flatten(),
        "Bootstrap particle filter w/ Cauchy dist.": fit.Q_t_given_t.flatten(),
    }
).plot(title="Filtering std")
