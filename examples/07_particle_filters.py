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
# # Particle filters
#
# Particle filters (sequential Monte Carlo) approximate the filtering distribution $X_t | Y_{1:t}$
# by a weighted sample of $N$ *particles*, propagated and resampled at every step.
# They apply to any state space model, at the price of Monte Carlo noise.
# yassm implements two proposals:
#
# - **bootstrap**: particles move according to the transition $p(x_t | x_{t-1})$,
#   and are weighted by the observation density $p(y_t | x_t)$;
# - **optimal** (locally adapted): particles move according to $p(x_t | x_{t-1}, y_t)$,
#   and are weighted by $p(y_t | x_{t-1})$,
#   which minimises the variance of the weights but is only tractable for some models, e.g. linear Gaussian ones.
#
# On a linear Gaussian model the Kalman filter is exact, so it can measure how good the particle filters are.
# This example uses the local level model of the Nile data from {doc}`01_kalman_filter`.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from example_data import load_nile, size
from numba import jit

from yassm import LinearGaussianParticleFilterModel, local_level_model

# %%
df = load_nile()
y = df.Nile.to_numpy(dtype=np.float64)
t = df.index.to_numpy()

# %% [markdown]
# ## The model
#
# {class}`~yassm.LinearGaussianParticleFilterModel` takes the same Numba functions as
# {class}`~yassm.LinearGaussianModel`, plus the distribution $N(μ_0, Γ_0)$ of $X_0$, here a vague prior
# centred on the data:

# %%
μ_0 = 0.5 * (y.min() + y.max())
Γ_0 = (y.max() - y.min()) ** 2

signature = "f8[:, ::1](f8[::1])"


@jit(signature, nopython=True, nogil=True, cache=True)
def identity(params):
    return np.ones((1, 1))


@jit(signature, nopython=True, nogil=True, cache=True)
def state_cov(params):
    return np.full((1, 1), params[1])


@jit(signature, nopython=True, nogil=True, cache=True)
def obs_cov(params):
    return np.full((1, 1), params[0])


@jit("Tuple((f8[::1], f8[:, ::1]))(f8[::1])", nopython=True, nogil=True, cache=True)
def initial_state(params):
    return np.full(1, μ_0), np.full((1, 1), Γ_0)


models = {
    method: LinearGaussianParticleFilterModel(
        2,
        identity,
        identity,
        state_cov,
        obs_cov,
        X0=initial_state,
        param_names=local_level_model.param_names,
        particle_filter_method=method,
    )
    for method in ("bootstrap", "optimal")
}

# %% [markdown]
# The Kalman filter, with the same prior, gives the MLE and the exact filtering distribution:

# %%
fit_kalman = local_level_model.fit(y, None, μ_0, Γ_0, mle_method="em", mle_n_iter=1000, t=t)
fit_kalman.params_to_series.to_frame("MLE")

# %% [markdown]
# ## Filtering
#
# With the parameters at the MLE, both particle filters track the Kalman filter.
# Particle filters run lazily: the first access to a result such as `m_t_given_t` runs the filter.

# %%
n_sample = size(10_000, fast=200)
fits = {method: model.fit(y, fit_kalman.params, n_sample=n_sample, t=t) for method, model in models.items()}

# %%
# %%time
for fit in fits.values():
    fit.m_t_given_t

# %%
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(t, y, ".", color="0.6", label="observations")
axes[0].plot(t, fit_kalman.m_t_given_t.ravel(), "k", label="Kalman filter")
for method, fit in fits.items():
    axes[0].plot(t, fit.m_t_given_t.ravel(), "--", label=f"{method} particle filter")
    axes[1].plot(t, (fit.m_t_given_t - fit_kalman.m_t_given_t).ravel(), label=method)
axes[0].set_title("Filtering mean")
axes[0].legend()
axes[1].set_title(f"Error of the particle filters with {n_sample} particles")
axes[1].legend()
plt.show()

# %% [markdown]
# ## Likelihood
#
# Particle filters also estimate the likelihood $p(y_{1:T})$, without bias.
# yassm's Kalman log-likelihood is conditional on the first observation (see `LinearGaussianFit.log_likelihood`),
# so the term $\log p(y_1)$, with $Y_1 \sim N(μ_0, Γ_0 + σ_ε^2)$, has to be added to compare them:

# %%
σ_ε_sq = fit_kalman.params[0]
log_p_y1 = -0.5 * (np.log(2.0 * np.pi * (Γ_0 + σ_ε_sq)) + (y[0] - μ_0) ** 2 / (Γ_0 + σ_ε_sq))
log_likelihood = fit_kalman.log_likelihood + log_p_y1

# %% [markdown]
# Repeating each particle filter a few times shows the Monte Carlo error of its log-likelihood,
# which shrinks as $1 / \sqrt{N}$:

# %%
# %%time
n_repeat = size(10, fast=2)
errors = pd.DataFrame(
    {
        (method, N): [
            model.fit(y, fit_kalman.params, n_sample=N).log_likelihood - log_likelihood for _ in range(n_repeat)
        ]
        for N in (size(1_000, fast=100), size(10_000, fast=200))
        for method, model in models.items()
    }
)
errors.agg(["mean", "std"]).T.rename_axis(["proposal", "particles"])

# %% [markdown]
# The optimal proposal roughly halves the error for the same number of particles.
# The gain is modest because the observations are noisy compared with the state
# ($σ_ε \approx 3 σ_Z$), so a single observation moves the particles little.
#
# ## Maximum likelihood by grid search
#
# The likelihood estimates are noisy, so gradient-based optimisers are of little use.
# A grid search over the parameters, with `mle_method="grid_search"`, is robust,
# and shows the shape of the likelihood:

# %%
n_grid = size(25, fast=4)
σ_εs, σ_Zs = np.meshgrid(np.linspace(80.0, 180.0, n_grid), np.linspace(10.0, 80.0, n_grid))
params_grid = np.square(np.column_stack((σ_εs.ravel(), σ_Zs.ravel())))

# %%
# %%time
fits_mle = {
    method: model.fit(
        y,
        None,
        mle_method="grid_search",
        mle_n_iter=1,
        mle_kwargs={"params_grid": params_grid},
        n_sample=size(2_000, fast=100),
        t=t,
    )
    for method, model in models.items()
}

# %%
log_likelihoods_kalman = fit_kalman.log_likelihoods(params_grid).reshape(n_grid, n_grid) + log_p_y1

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
levels = log_likelihood - np.array([20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.0])
panels = [("Kalman filter (exact)", log_likelihoods_kalman)] + [
    (f"{method} particle filter", -fit.neg_log_likelihoods_grid.reshape(n_grid, n_grid))
    for method, fit in fits_mle.items()
]
for ax, (title, ll) in zip(axes, panels, strict=True):
    cs = ax.contourf(σ_εs, σ_Zs, ll, levels=np.sort(levels), extend="both", cmap="viridis")
    ax.plot(*np.sqrt(fit_kalman.params), "rx")
    ax.set_title(title)
    ax.set_xlabel("σ_ε")
axes[0].set_ylabel("σ_Z")
fig.colorbar(cs, ax=axes, label="log-likelihood")
plt.show()

# %%
pd.DataFrame(
    {"Kalman filter": fit_kalman.params} | {f"{method} grid search": fit.params for method, fit in fits_mle.items()},
    index=local_level_model.param_names,
).apply(np.sqrt).rename(index={"obs_cov": "σ_ε", "state_cov": "σ_Z"})

# %% [markdown]
# Both surfaces have the right shape, but the likelihood is flat near its maximum, especially along $σ_Z$,
# so the Monte Carlo noise moves the grid-search MLE by several grid steps.
# More particles, or smoothing the estimated surface before taking its maximum, reduce this.
