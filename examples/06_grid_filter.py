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
# # Grid filter for a non-Gaussian model
#
# The Kalman filter is exact only for linear Gaussian models.
# For a one-dimensional state, any model can instead be filtered on a grid:
# {class}`~yassm.GridFilterModel` discretises the state space into $G$ points
# and propagates the filtering and smoothing distributions on them exactly,
# at a cost of $O(G^2)$ per time step.
#
# This example follows a level that is constant most of the time but occasionally jumps:
#
# $$
# \begin{aligned}
# Y_t &= X_t + ε_t, & ε_t &\sim N(0, σ_ε^2), \\
# X_t &= X_{t-1} + Z_t, & Z_t &\sim (1 - α)\, N(0, δ) + α\, N(0, σ_Z^2),
# \end{aligned}
# $$
#
# i.e. with probability $α$ the level jumps by a $N(0, σ_Z^2)$ amount,
# and otherwise it stays put (up to a tiny $δ$, which keeps the transition density smooth on the grid).
# A Gaussian model would have to choose between tracking the jumps and smoothing the noise.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from example_data import load_level_shifts, size
from numba import jit

from yassm import GridFilterModel, local_level_model
from yassm.lib.grid.one_dimension import prob_normal

# %%
df = load_level_shifts()
y = df.y.to_numpy()
ax = df.plot(title="Level shifts observed with noise", style=".", markersize=2)

# %% [markdown]
# ## The model
#
# A grid model is defined by two Numba-compiled functions of a pair of points, the parameters,
# and the grid spacing $Δx$:
#
# - the transition density $p(x_t | x_{t-1})$, and
# - the observation density $p(y_t | x_t)$.
#
# `prob_normal(x, μ, σ², Δx)` is the normal density times $Δx$, i.e. a probability on the grid.
# The noise variance $σ_ε^2 = 25$ is assumed known here, and the parameters are $(σ_Z^2, α)$:

# %%
δ = 1e-5
σ_ε_sq = 25.0


@jit("f8(f8, f8, f8[::1], f8)", nopython=True, nogil=True, cache=True)
def transition(x_t, x_tMinus1, params, Δx):
    σ_Z_sq, α = params
    return (1.0 - α) * prob_normal(x_t, x_tMinus1, δ, Δx) + α * prob_normal(x_t, x_tMinus1, σ_Z_sq, Δx)


@jit("f8(f8, f8, f8[::1], f8)", nopython=True, nogil=True, cache=True)
def observation(y_t, x_t, params, Δx):
    return prob_normal(y_t, x_t, σ_ε_sq, Δx)


# %% [markdown]
# The grid spans the range of the data with a margin on both sides:

# %%
G = size(100, fast=20)
y_mid = 0.5 * (y.min() + y.max())
y_half_range = 0.5 * (y.max() - y.min())
x_min, x_max = y_mid - 1.5 * y_half_range, y_mid + 1.5 * y_half_range

model = GridFilterModel(2, transition, observation, x_min, x_max, G, param_names=["σ_Z²", "α"])

# %% [markdown]
# ## Maximum likelihood
#
# `fit` takes the initial parameters and the distribution of $X_0$ on the grid, here uniform.
# The likelihood is maximised with `scipy.optimize.minimize`:

# %%
# %%time
fit = model.fit(
    y,
    np.array([0.1, 0.01]),
    np.full(G, 1.0 / G),
    mle_n_iter=1,
    mle_kwargs={"bounds": ((1e-8, None), (0.0, 1.0))},
)
fit.params_to_series.to_frame("MLE")

# %% [markdown]
# A jump happens on average every $1/α$ steps, with a typical size of $σ_Z$:

# %%
σ_Z_sq, α = fit.params
pd.Series({"σ_Z": np.sqrt(σ_Z_sq), "steps between jumps": 1.0 / α})

# %% [markdown]
# ## Filtering and smoothing
#
# The smoothed mean `m_t_given_T` follows the jumps sharply while staying flat in between.
# A local level model, the Gaussian counterpart, has to compromise:
# it wiggles with the noise between the jumps.

# %%
fit_gaussian = local_level_model.fit(y, None, 0.0, np.inf, mle_method="em", mle_n_iter=size(500, fast=10))

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(y, ".", markersize=2, color="0.6", label="observations")
ax.plot(fit_gaussian.m_t_given_T, label="local level model (Kalman smoother)")
ax.plot(fit.m_t_given_T, label="jump model (grid smoother)")
ax.legend()
ax.set_title("Smoothed level")
plt.show()

# %% [markdown]
# The full smoothing distribution on the grid is available as `p_t_given_T`, of shape $(T, G)$.
# It is concentrated within a few grid cells, of width $Δx$, and moves abruptly at the jumps:

# %%
t_jump = int(np.argmax(np.abs(np.diff(fit.m_t_given_T))))
window = slice(max(t_jump - 50, 0), t_jump + 50)
fig, ax = plt.subplots(figsize=(9, 4))
ax.imshow(
    fit.p_t_given_T[window].T,
    origin="lower",
    aspect="auto",
    extent=(window.start, window.stop, fit.x_array[0], fit.x_array[-1]),
    cmap="Blues",
)
ax.plot(range(window.start, window.stop), y[window], ".", color="C1", markersize=3)
ax.set_ylim(y[window].min() - 10, y[window].max() + 10)
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_title("Smoothing distribution around the largest jump")
plt.show()
