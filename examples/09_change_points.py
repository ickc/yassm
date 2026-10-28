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
# # Change points with particle filters
#
# Many series are piecewise simple: constant, linear, or with a constant variance, between rare change points.
# A state space model captures this with a transition noise that is *exactly zero* most of the time:
#
# $$
# U_t = \begin{cases} \mathrm{Cauchy}(0, 1) & \text{with probability } α, \\ 0 & \text{otherwise,} \end{cases}
# $$
#
# so that the state stays put except for occasional jumps of any size.
# Such a transition has no density, which rules out the grid filter of {doc}`06_grid_filter`,
# but a bootstrap particle filter only needs to *sample* it.
#
# This example fits such models to three simulated series, estimating the jump probability $α$ by grid search.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from example_data import load_level_shifts, load_trend_breaks, load_volatility_regimes, size
from numba import jit

from yassm import NonLinearParticleFilterModel
from yassm.lib.linear_gaussian import prob_normal_1d

# %% [markdown]
# The jump noise is shared by the three models, whose only parameter is $α$:


# %%
@jit("f8[::1](f8[::1], i8, i8)", nopython=True, nogil=True, cache=True)
def sampling_jumps(params, t, N):
    α = params[0]
    u_t = np.zeros(N)
    for i in range(N):
        if np.random.uniform(0.0, 1.0) < α:
            u_t[i] = np.random.standard_cauchy()
    return u_t


def make_model(pdf_Y_inplace, K_t_inplace, sampling_X0_tilde_inplace, state_to_obs, d=1):
    return NonLinearParticleFilterModel(
        1,
        pdf_Y_inplace=pdf_Y_inplace,
        K_t_inplace=K_t_inplace,
        sampling_X0_tilde_inplace=sampling_X0_tilde_inplace,
        sampling_U_t=sampling_jumps,
        state_to_obs=state_to_obs,
        param_names=["α"],
        d=d,
    )


# %% [markdown]
# The likelihood is estimated on a logarithmic grid of $α$ from $10^{-8}$ to 1,
# then on a finer grid around the best value:


# %%
def fit_α(model, y):
    n_sample = size(10_000, fast=100)
    n_grid = size(50, fast=4)
    αs = np.geomspace(1e-8, 1.0, n_grid)
    log_likelihoods = {}
    for _ in range(2):
        fit = model.fit(
            y,
            None,
            mle_method="grid_search",
            mle_n_iter=1,
            mle_kwargs={"params_grid": αs.reshape(-1, 1)},
            n_sample=n_sample,
        )
        log_likelihoods |= dict(zip(αs, -fit.neg_log_likelihoods_grid.ravel(), strict=True))
        α = fit.params[0]
        αs = np.geomspace(α / 10.0, min(α * 10.0, 1.0), n_grid)
    return model.fit(y, fit.params, n_sample=n_sample), pd.Series(log_likelihoods).sort_index()


def plot_log_likelihoods(log_likelihoods, α):
    ax = log_likelihoods.plot(logx=True, style=".-", xlabel="α", ylabel="log-likelihood")
    ax.axvline(α, color="k", linestyle="--")
    ax.set_ylim(log_likelihoods.max() - 50.0, log_likelihoods.max() + 5.0)
    plt.show()


# %% [markdown]
# ## Level shifts
#
# The data of {doc}`06_grid_filter`: a constant level with occasional shifts, observed with $N(0, 5^2)$ noise.
# The state is the level, $X_t = X_{t-1} + U_t$.

# %%
y = load_level_shifts().y.to_numpy()
μ_0 = y[:100].mean()


@jit("(f8[::1], f8[::1], f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def pdf_level(pdf, y_t, x, params, t):
    pdf[:] = prob_normal_1d(y_t[0], x[:, 0], 1.0 / 25.0, False)


@jit("(f8[:, ::1], f8[:, ::1], f8[::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def random_walk(x_t, x_tMinus1, u_t, params, t):
    x_t[:, 0] = x_tMinus1[:, 0] + u_t


@jit("(f8[:, ::1], f8[::1])", nopython=True, nogil=True, cache=True)
def x0_level(x_0, params):
    x_0[:, 0] = np.random.normal(μ_0, 5.0, x_0.shape[0])


@jit("f8[:, :, ::1](f8[::1], f8[:, :, ::1], i8)", nopython=True, nogil=True, cache=True)
def identity(params, x, t0):
    return x


# %%
# %%time
fit, log_likelihoods = fit_α(make_model(pdf_level, random_walk, x0_level, identity), y)
plot_log_likelihoods(log_likelihoods, fit.params[0])
f"α = {fit.params[0]:.2g}: a jump every {1.0 / fit.params[0]:.0f} steps on average"

# %% [markdown]
# Over 2048 observations, the likelihood estimates are noisy by several units, but the maximum is clear.
# The grid filter of {doc}`06_grid_filter` estimated a jump rate about four times lower:
# a jump of the standard Cauchy distribution is usually small compared with the shifts here, of order 10,
# so this model needs more, mostly invisible, jumps.

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(y, ".", markersize=2, color="0.6", label="observations")
ax.plot(fit.m_t_given_t[:, 0], label="filtering mean")
ax.legend()
ax.set_title("Level shifts")
plt.show()

# %% [markdown]
# ## Trend breaks
#
# A piecewise linear trend observed with $N(0, 2^2)$ noise.
# With the state $X_t = (L_t, L_{t-1})$, the local linear trend $L_t = 2 L_{t-1} - L_{t-2} + U_t$
# continues the current slope, until a jump $U_t$ changes both the level and the slope.

# %%
y = load_trend_breaks().y.to_numpy()
y_0 = y[0]
A_T = np.ascontiguousarray(np.array([[2.0, -1.0], [1.0, 0.0]]).T)


@jit("(f8[::1], f8[::1], f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def pdf_trend(pdf, y_t, x, params, t):
    pdf[:] = prob_normal_1d(y_t[0], np.ascontiguousarray(x[:, 0]), 1.0 / 4.0, False)


@jit("(f8[:, ::1], f8[:, ::1], f8[::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def local_linear_trend(x_t, x_tMinus1, u_t, params, t):
    x_t[:] = x_tMinus1 @ A_T
    x_t[:, 0] += u_t


@jit("(f8[:, ::1], f8[::1])", nopython=True, nogil=True, cache=True)
def x0_trend(x_0, params):
    x_0[:] = np.random.normal(y_0, 10.0, x_0.shape)


@jit("f8[:, :, ::1](f8[::1], f8[:, :, ::1], i8)", nopython=True, nogil=True, cache=True)
def level(params, x, t0):
    size, N, d = x.shape
    return np.ascontiguousarray(x[:, :, 0]).reshape(size, N, 1)


# %%
# %%time
fit, log_likelihoods = fit_α(make_model(pdf_trend, local_linear_trend, x0_trend, level, d=2), y)
plot_log_likelihoods(log_likelihoods, fit.params[0])
f"α = {fit.params[0]:.2g}"

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(y, ".", color="0.6", label="observations")
ax.plot(fit.m_t_given_t[:, 0], label="filtering mean")
ax.legend()
ax.set_title("Trend breaks")
plt.show()

# %% [markdown]
# ## Volatility regimes
#
# A zero-mean series whose variance changes between regimes: a stochastic volatility model
#
# $$Y_t | X_t \sim N(0, e^{X_t}), \qquad X_t = X_{t-1} + U_t,$$
#
# where the state is the log-variance.

# %%
y = load_volatility_regimes().y.to_numpy()


@jit("(f8[::1], f8[::1], f8[:, ::1], f8[::1], i8)", nopython=True, nogil=True, cache=True)
def pdf_volatility(pdf, y_t, x, params, t):
    # the precision e^{-x} overflows for very negative log-variances, where the density of any y_t ≠ 0 vanishes
    cov_inv = np.exp(-x[:, 0])
    pdf[:] = np.where(np.isposinf(cov_inv), 0.0, prob_normal_1d(y_t[0], 0.0, cov_inv, False))


@jit("(f8[:, ::1], f8[::1])", nopython=True, nogil=True, cache=True)
def x0_volatility(x_0, params):
    x_0[:, 0] = np.random.normal(0.0, 10.0, x_0.shape[0])


@jit("f8[:, :, ::1](f8[::1], f8[:, :, ::1], i8)", nopython=True, nogil=True, cache=True)
def zero_mean(params, x, t0):
    size, N, d = x.shape
    return np.zeros((size, N, 1))


# %%
# %%time
fit, log_likelihoods = fit_α(make_model(pdf_volatility, random_walk, x0_volatility, zero_mean), y)
plot_log_likelihoods(log_likelihoods, fit.params[0])
f"α = {fit.params[0]:.2g}"

# %% [markdown]
# Since $E[Y_t^2 | X_t] = e^{X_t}$, the filtered log-variance can be compared with $\log Y_t^2$,
# which scatters around it (shifted down by $E[\log χ^2_1] \approx -1.27$):

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(np.log(np.square(y)), ".", color="0.6", label="log y²")
ax.plot(fit.m_t_given_t[:, 0], label="filtering mean of the log-variance")
ax.legend()
ax.set_title("Volatility regimes")
plt.show()
