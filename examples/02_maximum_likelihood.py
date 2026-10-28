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
# # Maximum likelihood estimation
#
# yassm offers several ways to maximise the likelihood of a linear Gaussian model:
#
# - `"em"`: the EM algorithm, in closed form for the built-in one-dimensional models;
# - `"gradient"`: gradient ascent on the log-likelihood, using its analytic gradient;
# - `"scipy_minimize"`: any optimiser of {func}`scipy.optimize.minimize` on the log-likelihood from the Kalman filter,
#   available for every model including user-defined ones.
#
# This example compares them on a simulated AR(1) process observed with noise,
# which yassm calls the *local ratio model*:
#
# $$
# \begin{aligned}
# Y_t &= X_t + ε_t, & ε_t &\sim N(0, σ_ε^2), \\
# X_t &= φ X_{t-1} + Z_t, & Z_t &\sim N(0, σ_Z^2).
# \end{aligned}
# $$

# %%
import numpy as np
import pandas as pd
import statsmodels.tsa.api as tsa
from example_data import load_ar1_plus_noise

from yassm import local_ratio_model

# %%
df = load_ar1_plus_noise()
y = df.y.to_numpy()
ax = df.plot(title="AR(1) observed with noise")

# %% [markdown]
# The parameters are $(σ_ε^2, σ_Z^2, φ)$, in that order:

# %%
local_ratio_model.param_names

# %% [markdown]
# ## EM and gradient ascent
#
# With `params=None`, the initial parameters are guessed from the data.
# The prior on $X_0$ is diffuse, i.e. infinitely wide.

# %%
μ_0 = np.zeros(1)
Γ_0 = np.full((1, 1), np.inf)

# %%
# %%time
fit_em = local_ratio_model.fit(y, None, μ_0, Γ_0, mle_method="em", mle_n_iter=1000, mle_return_chain=True)

# %%
# %%time
fit_gradient = local_ratio_model.fit(y, None, μ_0, Γ_0, mle_method="gradient", mle_n_iter=1000, mle_return_chain=True)

# %%
ax = fit_em.mle_params_chain_to_frame.plot(logx=True, logy=True, title="EM iterations")

# %%
ax = fit_gradient.mle_params_chain_to_frame.plot(logx=True, logy=True, title="Gradient ascent iterations")

# %% [markdown]
# ## `scipy.optimize`
#
# `mle_kwargs` are passed to {func}`scipy.optimize.minimize`.
# The variances must be bounded below by zero:

# %%
# %%time
fit_scipy = local_ratio_model.fit(
    y,
    None,
    μ_0,
    Γ_0,
    mle_method="scipy_minimize",
    mle_n_iter=1,
    mle_return_chain=True,
    mle_kwargs={"method": "L-BFGS-B", "bounds": ((0.0, None), (0.0, None), (None, None))},
)

# %% [markdown]
# The three methods agree:

# %%
fits = {"EM": fit_em, "gradient": fit_gradient, "scipy.optimize": fit_scipy}
df_mle = pd.DataFrame([fit.params for fit in fits.values()], columns=local_ratio_model.param_names, index=fits)
df_mle["log-likelihood"] = [fit.log_likelihood for fit in fits.values()]
df_mle["iterations"] = [len(fit.mle_params_chain) - 1 for fit in fits.values()]
df_mle

# %% [markdown]
# The quasi-Newton optimiser needs far fewer iterations than EM or gradient ascent, each of which runs the
# Kalman filter (and, for EM, the smoother) once per iteration.
# EM never decreases the likelihood, needs no step size, and remains a robust default.
#
# ## Standard errors
#
# The inverse Hessian of the negative log-likelihood at the MLE, computed numerically with `numdifftools`,
# approximates the covariance of the estimates
# (equivalently, the covariance of a normal approximation to the posterior under a flat prior):

# %%
pd.DataFrame(
    {"MLE": fit_scipy.params, "standard error": np.sqrt(np.diag(fit_scipy.posterior_normal_approximation_cov))},
    index=local_ratio_model.param_names,
)

# %% [markdown]
# The smoothed state, with 95% intervals with and without the observation noise, and a 50-step forecast,
# which decays geometrically to 0 at rate $φ$:

# %%
fig, axes = fit_em.plot(T_forecast=50)

# %% [markdown]
# ## Comparison with `statsmodels`
#
# In `statsmodels`, this is an unobserved components model with an irregular and an AR(1) component:

# %%
res_sm = tsa.UnobservedComponents(y, "irregular", autoregressive=1).fit(disp=False)
pd.DataFrame(
    {"yassm": fit_em.params, "statsmodels": res_sm.params},
    index=[f"{a} / {b}" for a, b in zip(local_ratio_model.param_names, res_sm.param_names, strict=True)],
)

# %% [markdown]
# The estimates are close but not identical, mostly in $σ_Z^2$:
# `statsmodels` initialises the stationary AR(1) component from its stationary distribution
# $N(0, σ_Z^2 / (1 - φ^2))$, which depends on the parameters,
# while yassm's prior on $X_0$ here is diffuse and fixed.
