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
# # Heavy-tailed noise
#
# The Nile series of {doc}`01_kalman_filter` is known for an abrupt drop around 1899, when the first Aswan dam
# was built. The Gaussian local level model can only follow it gradually: a Gaussian random walk moves in small steps.
# Heavy-tailed noise allows rare large steps.
# This example compares, by their likelihoods, the Gaussian local level model with two heavy-tailed variants:
#
# $$
# \begin{aligned}
# Y_t &= X_t + τ_ε V_t, \\
# X_t &= X_{t-1} + τ_Z U_t, & U_t &\sim \mathrm{Cauchy}(0, 1),
# \end{aligned}
# $$
#
# with $V_t \sim \mathrm{Cauchy}(0, 1)$ (Cauchy noise everywhere) or $V_t \sim N(0, 1)$ (Cauchy noise on the level only).
# They are no longer linear Gaussian models, so they are filtered with a bootstrap particle filter,
# {class}`~yassm.NonLinearParticleFilterModel`.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from example_data import load_nile, size
from numba import jit

from yassm import NonLinearParticleFilterModel, local_level_model
from yassm.lib.linear_gaussian import prob_cauchy, prob_normal_1d

# %%
df = load_nile()
y = df.Nile.to_numpy(dtype=np.float64)
t = df.index.to_numpy()

# %% [markdown]
# ## The models
#
# A non-linear model is defined by Numba functions that act on all $N$ particles at once,
# each writing its result in place:
#
# - `pdf_Y_inplace(pdf, y_t, x, params, t)`: the observation density $p(y_t | x_t)$ of each particle;
# - `K_t_inplace(x_t, x_tMinus1, u_t, params, t)`: the transition $X_t = K_t(X_{t-1}, U_t)$;
# - `sampling_U_t(params, t, N)`: draws of the transition noise $U_t$;
# - `sampling_X0_tilde_inplace(x_0, params)`: draws of the initial state;
# - `state_to_obs(params, x, t0)`: the observation implied by a state, for plotting and forecasting.
#
# The states have shape $(N, d)$, here with $d = 1$, and the parameters are $(τ_ε^2, τ_Z^2)$.
# `prob_cauchy(x, x_0, τ², normalise)` is the Cauchy density of scale $τ$,
# and `prob_normal_1d(x, μ, 1 / σ², normalise)` the normal density.

# %%
scale = y.max()


@jit("(f8[::1], f8[::1], f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def pdf_Y_cauchy(pdf, y_t, x, params, t):
    pdf[:] = prob_cauchy(y_t[0], x[:, 0], params[0], False)


@jit("(f8[::1], f8[::1], f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def pdf_Y_normal(pdf, y_t, x, params, t):
    pdf[:] = prob_normal_1d(y_t[0], x[:, 0], 1.0 / params[0], False)


@jit("(f8[:, ::1], f8[:, ::1], f8[::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def K_t_inplace(x_t, x_tMinus1, u_t, params, t):
    x_t[:, 0] = x_tMinus1[:, 0] + u_t


@jit("f8[::1](f8[::1], i8, i8)", nopython=True, nogil=True, cache=True)
def sampling_U_t(params, t, N):
    return np.sqrt(params[1]) * np.random.standard_cauchy(N)


@jit("(f8[:, ::1], f8[::1])", nopython=True, nogil=True, cache=True)
def sampling_X0_tilde_inplace(x_0, params):
    x_0[:, 0] = np.random.normal(0.0, scale, x_0.shape[0])


@jit("f8[:, :, ::1](f8[::1], f8[:, :, ::1], i8)", nopython=True, nogil=True, cache=True)
def state_to_obs(params, x, t0):
    return x


models = {
    name: NonLinearParticleFilterModel(
        2,
        pdf_Y_inplace=pdf_Y,
        K_t_inplace=K_t_inplace,
        sampling_X0_tilde_inplace=sampling_X0_tilde_inplace,
        sampling_U_t=sampling_U_t,
        state_to_obs=state_to_obs,
        param_names=["τ_ε²", "τ_Z²"],
    )
    for name, pdf_Y in (("Cauchy everywhere", pdf_Y_cauchy), ("Cauchy level", pdf_Y_normal))
}

# %% [markdown]
# ## Maximum likelihood
#
# As in {doc}`07_particle_filters`, the likelihood estimates are noisy,
# so the MLE is found by grid search: first coarsely over a wide, logarithmic range,
# then finely around the best point.


# %%
def grid_search(model, τ_εs, τ_Zs, n_sample):
    τ_ε_grid, τ_Z_grid = np.meshgrid(τ_εs, τ_Zs)
    params_grid = np.square(np.column_stack((τ_ε_grid.ravel(), τ_Z_grid.ravel())))
    return model.fit(
        y, None, mle_method="grid_search", mle_n_iter=1, mle_kwargs={"params_grid": params_grid}, n_sample=n_sample, t=t
    )


def mle(model):
    n_grid = size(20, fast=4)
    coarse = grid_search(
        model, np.geomspace(5.0, 500.0, n_grid), np.geomspace(0.05, 200.0, n_grid), size(2_000, fast=100)
    )
    τ_ε, τ_Z = np.sqrt(coarse.params)
    fine = grid_search(
        model,
        np.linspace(0.5 * τ_ε, 2.0 * τ_ε, n_grid),
        np.linspace(0.25 * τ_Z, 4.0 * τ_Z, n_grid),
        size(20_000, fast=100),
    )
    return fine.params


# %%
# %%time
params = {name: mle(model) for name, model in models.items()}
pd.DataFrame(params, index=["τ_ε²", "τ_Z²"]).apply(np.sqrt).rename(index={"τ_ε²": "τ_ε", "τ_Z²": "τ_Z"})

# %% [markdown]
# ## Model comparison
#
# The local level model with the same vague prior on $X_0$ serves as the Gaussian reference.
# Its Kalman log-likelihood is conditional on $y_1$, so $\log p(y_1)$ is added to compare it with the particle filters':

# %%
Γ_0 = scale**2
fit_gaussian = local_level_model.fit(y, None, 0.0, Γ_0, mle_method="em", mle_n_iter=1000, t=t)
σ_ε_sq = fit_gaussian.params[0]
log_p_y1 = -0.5 * (np.log(2.0 * np.pi * (Γ_0 + σ_ε_sq)) + y[0] ** 2 / (Γ_0 + σ_ε_sq))

# %%
n_sample = size(100_000, fast=200)
fits = {name: model.fit(y, params[name], n_sample=n_sample, t=t) for name, model in models.items()}
pd.Series(
    {"Gaussian": fit_gaussian.log_likelihood + log_p_y1} | {name: fit.log_likelihood for name, fit in fits.items()},
    name="log-likelihood at the MLE",
)

# %% [markdown]
# All three models have two parameters, so their likelihoods compare directly.
# Cauchy noise everywhere fits worse than the Gaussian model: the observation noise is not heavy-tailed,
# and the fit makes the level chase the observations.
# Cauchy noise on the level alone fits best. Its level is much smoother than the Gaussian one,
# except for a large drop around 1899:

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, y, ".", color="0.6", label="observations")
ax.plot(t, fit_gaussian.m_t_given_t.ravel(), label="Gaussian (Kalman filter)")
for name, fit in fits.items():
    ax.plot(t, fit.m_t_given_t.ravel(), label=f"{name} (particle filter)")
ax.axvline(np.datetime64("1899"), color="k", linewidth=0.5, linestyle="--")
ax.legend()
ax.set_title("Filtering mean of the level")
plt.show()

# %% [markdown]
# A model with explicit change points, such as the mixture of {doc}`06_grid_filter`, is another way to capture such jumps;
# {doc}`09_change_points` combines that idea with particle filters.
