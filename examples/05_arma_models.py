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
# # ARMA models: exact posteriors and multimodal likelihoods
#
# This example treats two small ARMA models in a Bayesian way:
# an AR(2) model of the yearly sunspot numbers, whose posterior can be sampled exactly,
# and an MA(1) model, written as a state space model,
# whose likelihood has two maxima.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from example_data import load_ma1, load_sunspots, size
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA

from yassm import moving_average_1_model
from yassm.lib.linear_gaussian.model.autoregressive_2 import direct_sampling, neg_log_likelihood

# %% [markdown]
# ## AR(2): sunspots
#
# The model is
#
# $$Y_t = φ_0 + φ_1 Y_{t-1} + φ_2 Y_{t-2} + Z_t, \quad Z_t \sim N(0, σ_Z^2).$$
#
# Conditional on the first two observations, its likelihood is that of a linear regression,
# so under a prior flat in $(φ_0, φ_1, φ_2, \log σ_Z)$ the posterior is normal–inverse-gamma
# and can be sampled exactly, without Markov chains.

# %%
df = load_sunspots()
y = df.SNvalue.to_numpy(copy=True)
ax = df.SNvalue.plot(title="Yearly mean total sunspot number")

# %%
names = ["σ_Z²", "φ_0", "φ_1", "φ_2"]
mle = minimize(neg_log_likelihood, [y.var(), 0.0, 1.0, 0.0], args=(y,), bounds=[(1e-8, None)] + [(None, None)] * 3).x

# %%
# %%time
samples = pd.DataFrame(direct_sampling(y, size(10_000, fast=100)), columns=names)

# %%
fig, axes = plt.subplots(1, len(names), figsize=(16, 3))
for ax, name, value in zip(axes, names, mle, strict=True):
    sns.histplot(samples[name], kde=True, ax=ax)
    ax.axvline(value, color="k", linestyle="--", label="MLE")
axes[0].legend()
plt.show()

# %% [markdown]
# `statsmodels` maximises the exact likelihood instead, and parametrises the constant by the mean $μ$ of the process,
# so $φ_0 = μ (1 - φ_1 - φ_2)$. After converting, the estimates agree within their posterior standard deviations:

# %%
res_sm = ARIMA(y, order=(2, 0, 0)).fit()
μ, φ_1, φ_2, σ_Z_sq = res_sm.params
pd.DataFrame(
    {
        "conditional MLE": mle,
        "posterior mean": samples.mean(),
        "posterior std": samples.std(),
        "statsmodels": [σ_Z_sq, μ * (1.0 - φ_1 - φ_2), φ_1, φ_2],
    },
    index=names,
)

# %% [markdown]
# The roots of $1 - φ_1 z - φ_2 z^2$ are complex, so the process is a damped oscillation,
# with a period of:

# %%
period = 2.0 * np.pi / np.arccos(samples.φ_1 / (2.0 * np.sqrt(-samples.φ_2)))
f"{period.mean():.1f} ± {period.std():.1f} years"

# %% [markdown]
# ## MA(1): two maxima
#
# The MA(1) model
#
# $$Y_t = ε_t + θ ε_{t-1}, \quad ε_t \sim N(0, σ^2)$$
#
# is a linear Gaussian state space model with state $X_t = (Y_t, θ ε_t)$:
#
# $$
# Y_t = \begin{pmatrix} 1 & 0 \end{pmatrix} X_t, \qquad
# X_t = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} X_{t-1} + \begin{pmatrix} 1 \\ θ \end{pmatrix} ε_t.
# $$
#
# `moving_average_1_model` starts $X_0$ from its stationary distribution, which depends on the parameters,
# so the Kalman filter gives the exact likelihood.
# The prior passed to `fit` is then ignored.

# %%
df = load_ma1()
y = df.y.to_numpy(copy=True)
ax = df.plot(title="Simulated MA(1)")

# %%
μ_0, Γ_0 = np.zeros(2), np.identity(2)  # ignored
fits = {θ_0: moving_average_1_model.fit(y, [y.var(), θ_0], μ_0, Γ_0, mle_n_iter=1) for θ_0 in (-0.5, -2.0)}
pd.DataFrame(
    [[*fit.params, fit.log_likelihood] for fit in fits.values()],
    columns=[*moving_average_1_model.param_names, "log-likelihood"],
    index=pd.Index(fits, name="initial θ"),
)

# %% [markdown]
# The two maxima have the same likelihood: $(σ^2, θ)$ and $(θ^2 σ^2, 1/θ)$ give the same autocovariances,
# hence the same Gaussian likelihood.
# Only the solution with $|θ| < 1$ is *invertible*, i.e. lets $ε_t$ be recovered from past observations;
# it is the one `statsmodels` reports:

# %%
fit, fit_twin = fits[-0.5], fits[-2.0]
σ_sq, θ = fit.params
θ_sm, σ_sq_sm = ARIMA(y, order=(0, 0, 1), trend="n").fit().params
pd.DataFrame(
    {"invertible MLE": fit.params, "twin": fit_twin.params, "(θ²σ², 1/θ)": [θ**2 * σ_sq, 1.0 / θ]},
    index=moving_average_1_model.param_names,
).assign(statsmodels=[σ_sq_sm, θ_sm])

# %% [markdown]
# ### Normal approximations
#
# `posterior_normal_approximation_cov` is the inverse of the negative Hessian of the log-likelihood
# at the MLE, i.e. the covariance of the normal (Laplace) approximation around each maximum.
# Compared with the likelihood itself, evaluated on a grid with `neg2_log_likelihoods`,
# each approximation captures its own mode but neither describes the whole likelihood:

# %%
n = size(100, fast=10)
σ_sqs, θs = np.meshgrid(np.linspace(0.45, 1.45, n), np.linspace(-1.6, -0.55, n))
grid = np.column_stack((σ_sqs.ravel(), θs.ravel()))

# %%
# %%time
log_likelihood = -0.5 * moving_average_1_model.neg2_log_likelihoods(grid, y.reshape(-1, 1), μ_0, Γ_0)
likelihood = np.exp(log_likelihood - fit.log_likelihood).reshape(n, n)


# %%
def laplace(fit) -> np.ndarray:
    """Normal approximation around the fit's MLE, normalised to a peak of 1."""
    d = grid - fit.params
    return np.exp(-0.5 * np.einsum("ij,jk,ik->i", d, fit.posterior_normal_approximation_cov_inv, d)).reshape(n, n)


levels = [0.05, 0.25, 0.5, 0.75, 0.95]
fig, ax = plt.subplots(figsize=(6, 5))
cs = ax.contourf(σ_sqs, θs, likelihood, levels=[0.0, *levels, 1.0], cmap="Blues")
for f, color in ((fit, "C1"), (fit_twin, "C3")):
    ax.contour(σ_sqs, θs, laplace(f), levels=levels, colors=color, linewidths=1)
    ax.plot(*f.params, "x", color=color)
fig.colorbar(cs, label="likelihood relative to the maximum")
ax.set_xlabel("σ²")
ax.set_ylabel("θ")
ax.set_title("Likelihood (filled) and normal approximations (lines)")
plt.show()

# %% [markdown]
# The standard errors from the approximation at the invertible maximum
# agree with those of `statsmodels` from its numerical Hessian:

# %%
res_sm = ARIMA(y, order=(0, 0, 1), trend="n").fit(cov_type="approx")
pd.DataFrame(
    {
        "yassm": np.sqrt(np.diag(fit.posterior_normal_approximation_cov)),
        "statsmodels": res_sm.bse[::-1],
    },
    index=moving_average_1_model.param_names,
).rename_axis(columns="standard error")
