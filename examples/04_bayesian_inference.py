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
# # Simulation smoothing and Gibbs sampling
#
# Beyond point estimates, yassm samples from
#
# - the smoothing distribution $X_{0:T} | Y_{1:T}, θ$ by *forward-filtering backward-sampling* (FFBS,
#   Carter & Kohn 1994; Frühwirth-Schnatter 1994), and
# - the joint posterior $X_{0:T}, θ | Y_{1:T}$ by Gibbs sampling, alternating FFBS with draws of $θ | X_{0:T}, Y_{1:T}$.
#
# For the built-in models the second step is conjugate,
# with a prior flat in $\log σ$ for every standard deviation and flat in $φ$ for the local ratio model.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from example_data import load_ar1_plus_noise, load_google_trends, load_nile, size

from yassm import local_level_model, local_linear_model, local_ratio_model


def plot_posteriors(samples: pd.DataFrame, mle: pd.Series) -> None:
    """Histograms of posterior samples, with the MLE marked."""
    fig, axes = plt.subplots(1, samples.shape[1], figsize=(4 * samples.shape[1], 3))
    for ax, name in zip(np.atleast_1d(axes), samples.columns, strict=True):
        sns.histplot(samples[name], kde=True, ax=ax)
        ax.axvline(mle[name], color="k", linestyle="--", label="MLE")
        ax.legend()
    plt.show()


# %% [markdown]
# ## Simulation smoothing
#
# The weekly Google Trends interest in "google" is modelled by a local linear trend.
# Passing `n_sample` to `fit` draws that many paths from the smoothing distribution at the MLE:

# %%
df = load_google_trends()
y = df.google.to_numpy()
t = df.index.to_numpy()
ax = df.plot(title='Google Trends interest in "google"')

# %%
# %%time
fit = local_linear_model.fit(
    y,
    None,
    μ_0=np.zeros(2),
    Γ_0=1e8 * np.identity(2),
    mle_n_iter=1,
    mle_kwargs={"bounds": ((0.0, None), (0.0, None))},
    n_sample=size(500, fast=50),
    t=t,
)
fit.params_to_series.to_frame("MLE")

# %% [markdown]
# The sample mean and variance of the paths converge to the smoothed moments of the Kalman smoother:

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, name, sample, exact in (
    (axes[0], "mean", fit.smoothed_obs_from_samples, fit.smoothed_obs),
    (axes[1], "variance", fit.smoothed_obs_cov_from_samples, fit.smoothed_obs_cov),
):
    ax.loglog(exact.ravel(), sample.ravel(), ".", alpha=0.5)
    ax.axline((exact.mean(), exact.mean()), (exact.max(), exact.max()), color="k", linewidth=0.5)
    ax.set_xlabel("Kalman smoother")
    ax.set_ylabel(f"{fit.n_sample} FFBS samples")
    ax.set_title(f"Smoothed {name}")
plt.show()

# %% [markdown]
# ## Gibbs sampling
#
# For the Nile data and the local level model, `gibbs_burn_in` discards the start of a chain of `n_sample` draws.
# The fit is initialised at the MLE (by EM here), so the chain starts in the bulk of the posterior:

# %%
df = load_nile()
y = df.Nile.to_numpy()

# %%
# %%time
n_kept = size(20_000, fast=200)
burn_in = size(2_000, fast=100)
fit = local_level_model.fit(
    y,
    None,
    μ_0=0.0,
    Γ_0=np.inf,
    mle_method="em",
    mle_n_iter=1000,
    n_sample=burn_in + n_kept,
    gibbs_burn_in=burn_in,
    t=df.index.to_numpy(),
)

# %% [markdown]
# `gibbs_params_chain` holds the parameter draws, here converted to standard deviations:

# %%
names = ["σ_ε", "σ_Z"]
samples = pd.DataFrame(np.sqrt(fit.gibbs_params_chain), columns=names)
mle = pd.Series(np.sqrt(fit.params), index=names)
pd.DataFrame({"MLE": mle, "posterior mean": samples.mean(), "posterior std": samples.std()})

# %%
plot_posteriors(samples, mle)

# %% [markdown]
# The smoothed level from the Gibbs samples accounts for the uncertainty in the parameters as well:

# %%
fig, axes = fit.plot(
    plot_smoothed_CI_with_obs_cov=False,
    plot_smoothed_from_gibbs_samples=True,
    plot_smoothed_from_gibbs_samples_CI=True,
)

# %% [markdown]
# ## A degenerate chain
#
# The prior flat in $\log σ$ is improper, and so can be the posterior:
# near $σ_Z = 0$ the likelihood of a variance component tends to a positive constant,
# so the posterior mass there is infinite (Gelman, 2006).
# Whether a finite chain notices depends on the data.
# For the Nile data above it does not, but for the AR(1) plus noise model of {doc}`02_maximum_likelihood`,
# whose AR(1) component is weak relative to the noise, the chain drifts to $σ_Z \to 0$ and stays there:

# %%
df = load_ar1_plus_noise()
y = df.y.to_numpy()

# %%
# %%time
n_kept = size(1_000, fast=20)
n_sample = size(200_000, fast=5_000)
fit = local_ratio_model.fit(
    y,
    None,
    μ_0=0.0,
    Γ_0=10.0,
    mle_method="em",
    mle_n_iter=1000,
    n_sample=n_sample,
    gibbs_burn_in=n_sample - n_kept,
)
names = ["σ_ε", "σ_Z", "φ"]
mle = pd.Series([*np.sqrt(fit.params[:2]), fit.params[2]], index=names)


def to_frame(chain: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.column_stack((np.sqrt(chain[:, :2]), chain[:, 2])), columns=names)


# %%
chain = to_frame(fit.gibbs_params_chain_with_burn_in)
axes = chain.plot(subplots=True, logx=True, figsize=(6, 6), title="Gibbs chain, including burn-in")
for ax, name in zip(axes, names, strict=True):
    ax.axhline(mle[name], color="k", linestyle="--")
plt.show()

# %%
samples = to_frame(fit.gibbs_params_chain)
pd.DataFrame({"MLE": mle, "posterior mean": samples.mean(), "posterior median": samples.median()})

# %% [markdown]
# The principled remedy is a proper prior that vanishes at $σ_Z = 0$, which the conjugate steps of the built-in
# models do not offer.
# A pragmatic one is to keep only the draws within a few standard deviations of the MLE,
# as measured by the log-likelihood (here within 1.6σ, i.e. $\log L > \log L_{\mathrm{MLE}} - 1.6^2/2$),
# which truncates the posterior to the neighbourhood of the MLE.
# `run_gibbs_sampling` re-runs the sampler with this filter;
# with only about 1% of the draws kept, the chain must be long:

# %%
# %%time
fit.run_gibbs_sampling(filter_n_std=1.6)
samples_filtered = to_frame(fit.gibbs_params_chain)
pd.DataFrame({"MLE": mle, "posterior mean": samples_filtered.mean(), "posterior median": samples_filtered.median()})

# %%
plot_posteriors(samples_filtered, mle)

# %%
fig, axes = fit.plot(
    plot_obs=False,
    plot_smoothed_CI_with_obs_cov=False,
    plot_smoothed_from_gibbs_samples=True,
    plot_smoothed_from_gibbs_samples_CI=True,
)
