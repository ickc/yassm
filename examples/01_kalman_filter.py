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
# # Kalman filtering and smoothing
#
# The flow of the river Nile is the classic example of the *local level model*
# (Durbin & Koopman, *Time Series Analysis by State Space Methods*): a random walk observed with noise,
#
# $$
# \begin{aligned}
# Y_t &= X_t + ε_t, & ε_t &\sim N(0, σ_ε^2), \\
# X_t &= X_{t-1} + Z_t, & Z_t &\sim N(0, σ_Z^2).
# \end{aligned}
# $$
#
# This example estimates $σ_ε^2$ and $σ_Z^2$ by maximum likelihood,
# extracts the level $X_t$ with the Kalman smoother, forecasts it,
# and fills in missing observations, cross-checking the results against `statsmodels`.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.api as tsa
from example_data import load_nile, load_nile_gaps

from yassm import local_level_model

# %%
df = load_nile()
y = df.Nile.to_numpy()
t = df.index.to_numpy()
ax = df.plot(title="Annual flow of the Nile at Aswan")
# %% [markdown]
# ## Maximum likelihood by EM
#
# `fit` takes the observations, initial parameters $(σ_ε^2, σ_Z^2)$ and the prior $X_0 \sim N(μ_0, Γ_0)$.
# An infinite $Γ_0$ is a *diffuse* prior: nothing is assumed about the initial level.
# With `params=None`, the initial parameters are guessed from the variances of the first and second differences of $Y$.
# The local level model implements the EM algorithm in closed form,
# here with 1000 iterations, keeping the chain of iterates to check convergence.

# %%
# %%time
fit = local_level_model.fit(y, None, μ_0=0.0, Γ_0=np.inf, mle_method="em", mle_n_iter=1000, mle_return_chain=True, t=t)

# %%
ax = fit.mle_params_chain_to_frame.plot(logx=True, logy=True, title="EM iterations")
# %%
fit.params_to_series.to_frame("MLE")

# %% [markdown]
# The fit holds the filtered ($X_t | Y_{1:t}$) and smoothed ($X_t | Y_{1:T}$) moments,
# e.g. `fit.m_t_given_t` and `fit.m_t_given_T`, all computed lazily on first access.
# `plot` shows the smoothed level with its 95% interval, with and without the observation noise,
# and a forecast of the next 10 years:

# %%
fig, axes = fit.plot(T_forecast=10)

# %% [markdown]
# ## Comparison with `statsmodels`

# %%
# %%time
res_sm = tsa.UnobservedComponents(df, "local level").fit(disp=False)

# %%
pd.DataFrame({"yassm": fit.params, "statsmodels": res_sm.params.to_numpy()}, index=fit.param_names)

# %% [markdown]
# The two estimates agree to within 1%, and so do the smoothed levels:

# %%
fig = res_sm.plot_components(which="smoothed")

# %% [markdown]
# ## Missing observations
#
# The Kalman filter handles missing observations (`NaN`) by skipping the update step.
# Here 15 observations of the Nile series are removed,
# and the smoother is used to predict them:

# %%
df_gaps = load_nile_gaps()
y_gaps = df_gaps.Nile.to_numpy()
missing = np.isnan(y_gaps)
f"{missing.sum()} of {missing.size} observations missing"

# %%
fit_gaps = local_level_model.fit(y_gaps, None, μ_0=0.0, Γ_0=np.inf, mle_method="em", mle_n_iter=1000, t=t)
fit_gaps.params_to_series.to_frame("MLE")

# %%
fig, axes = fit_gaps.plot()

# %% [markdown]
# `smoothed_obs_CI` returns the lower bound, mean and upper bound of $Y_t | Y_{1:T}$,
# which, compared with the observations that were removed, are:

# %%
lower, mean, upper = fit_gaps.smoothed_obs_CI(α=0.05, with_obs_cov=True)
df_missing = pd.DataFrame(
    {"actual": y[missing], "predicted": mean[missing, 0], "lower": lower[missing, 0], "upper": upper[missing, 0]},
    index=df.index[missing],
)
df_missing

# %%
fig, ax = plt.subplots()
ax.errorbar(
    df_missing.index,
    df_missing.predicted,
    [df_missing.predicted - df_missing.lower, df_missing.upper - df_missing.predicted],
    fmt="o",
    capsize=5,
    label="predicted, 95% interval",
)
ax.plot(df_missing.index, df_missing.actual, "x", label="actual")
ax.legend()
ax.set_title("Predicted missing observations")
plt.show()

# %% [markdown]
# Only the unusually low flow of 1913 falls outside its interval:

# %%
f"{((df_missing.lower <= df_missing.actual) & (df_missing.actual <= df_missing.upper)).mean():.0%} of the removed observations lie in the 95% interval"
