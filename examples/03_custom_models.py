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
# # Trend extraction and custom models
#
# This example recovers a smooth signal from noisy observations,
# first with the built-in *local linear trend* model,
# then with a custom model that adds an AR(1) component,
# defined in a few lines with {class}`~yassm.LinearGaussianModel`.
#
# Both datasets are simulated around the same smooth function $f$ on $u \in [0, 1)$,
# so the estimates can be compared with the truth.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.api as tsa
from example_data import load_smooth_signal, load_smooth_signal_with_ar1, smooth_signal_truth
from numba import jit

from yassm import LinearGaussianModel, local_linear_model

# %% [markdown]
# ## Local linear trend
#
# The local linear trend model is a second-order random walk observed with noise,
#
# $$
# \begin{aligned}
# Y_t &= X_t + ε_t, & ε_t &\sim N(0, σ_ε^2), \\
# X_t &= 2 X_{t-1} - X_{t-2} + Z_t, & Z_t &\sim N(0, σ_Z^2),
# \end{aligned}
# $$
#
# whose smoothed state is a cubic smoothing spline (Wahba, 1978).
# Its state vector is $(X_t, X_{t-1})$.

# %%
df = load_smooth_signal()
y = df.y.to_numpy()
u = np.arange(y.size) / y.size
f = smooth_signal_truth(u)

# %% [markdown]
# `local_linear_model` is a {class}`~yassm.LinearGaussianModel`, fitted with `scipy.optimize.minimize` by default:

# %%
# %%time
μ_0 = np.zeros(2)
Γ_0 = 1e8 * np.identity(2)
fit = local_linear_model.fit(
    y, None, μ_0, Γ_0, mle_n_iter=1, mle_kwargs={"method": "L-BFGS-B", "bounds": ((0.0, None), (0.0, None))}, t=u
)

# %%
res_sm = tsa.UnobservedComponents(y, "smooth trend").fit(disp=False)
pd.DataFrame({"yassm": fit.params, "statsmodels": res_sm.params}, index=fit.param_names)

# %% [markdown]
# The smoothed state with its 95% interval follows the true signal closely:

# %%
fig, axes = fit.plot(plot_obs=False, plot_smoothed_CI_with_obs_cov=False)
axes[0].plot(u, f, "k--", label="truth")
axes[0].legend()
axes[0].set_title("Smoothed state")
plt.show()

# %% [markdown]
# ## A custom model: trend plus AR(1)
#
# The second dataset adds an AR(1) component $A_t$ to the signal.
# A model for it is a smooth trend (level $L_t$ and slope $S_t$) plus the AR(1) component and noise:
#
# $$
# \begin{aligned}
# Y_t &= L_t + A_t + ε_t, & ε_t &\sim N(0, σ_ε^2), \\
# L_t &= L_{t-1} + S_{t-1}, \\
# S_t &= S_{t-1} + Z_t, & Z_t &\sim N(0, σ_S^2), \\
# A_t &= φ A_{t-1} + W_t, & W_t &\sim N(0, σ_A^2).
# \end{aligned}
# $$
#
# {class}`~yassm.LinearGaussianModel` describes a model $Y_t = B X_t + V_t$, $X_t = A X_{t-1} + U_t$,
# with $U_t \sim N(0, Σ)$ and $V_t \sim N(0, R)$,
# by Numba-compiled functions mapping the parameters to $A$, $B$, $Σ$ and $R$.
# Here the state is $X_t = (L_t, S_t, A_t)$ and the parameters are $(σ_ε^2, σ_S^2, σ_A^2, φ)$:

# %%
signature = "f8[:, ::1](f8[::1])"


@jit(signature, nopython=True, nogil=True, cache=True)
def transition(params):
    return np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, params[3]]])


@jit(signature, nopython=True, nogil=True, cache=True)
def design(params):
    return np.array([[1.0, 0.0, 1.0]])


@jit(signature, nopython=True, nogil=True, cache=True)
def state_cov(params):
    out = np.zeros((3, 3))
    out[1, 1] = params[1]
    out[2, 2] = params[2]
    return out


@jit(signature, nopython=True, nogil=True, cache=True)
def obs_cov(params):
    return np.full((1, 1), params[0])


# forecast the trend alone, without the AR(1) component
@jit(signature, nopython=True, nogil=True, cache=True)
def design_forecast(params):
    return np.array([[1.0, 0.0, 0.0]])


trend_ar1_model = LinearGaussianModel(
    4,
    transition,
    design,
    state_cov,
    obs_cov,
    B0_forecast=design_forecast,
    param_names=["obs_cov", "slope_cov", "ar_cov", "φ"],
)

# %% [markdown]
# The parameters have very different scales: the slope variance is of order $10^{-5}$, the others of order 1.
# Quasi-Newton optimisers with finite-difference gradients, such as SciPy's default L-BFGS-B,
# tend to stop early on such problems, while the derivative-free Nelder–Mead method copes better.
# All three start from the same point, with the slope variance guessed from the first model:

# %%
df = load_smooth_signal_with_ar1()
y = df.y.to_numpy()
μ_0 = np.zeros(3)
Γ_0 = 1e8 * np.identity(3)
bounds = ((0.0, None), (0.0, None), (0.0, None), (-1.0, 1.0))
params0 = [0.5, 1e-5, 0.5, 0.5]

# %%
# %%time
methods = ("L-BFGS-B", "Powell", "Nelder-Mead")
fits = {
    method: trend_ar1_model.fit(
        y, params0, μ_0, Γ_0, mle_n_iter=1, mle_kwargs={"method": method, "bounds": bounds}, t=u
    )
    for method in methods
}
pd.DataFrame(
    [[*fit.params, fit.log_likelihood] for fit in fits.values()],
    columns=[*trend_ar1_model.param_names, "log-likelihood"],
    index=pd.Index(methods, name="method"),
)

# %%
fit = fits["Nelder-Mead"]

# %% [markdown]
# The smoothed level $L_t$ is the first component of the smoothed state `m_t_given_T`,
# with variance `Q_t_given_T[:, 0, 0]`:

# %%
level = fit.m_t_given_T[:, 0]
level_std = np.sqrt(fit.Q_t_given_T[:, 0, 0])

fig, ax = plt.subplots()
ax.plot(u, y, ".", alpha=0.3, label="observations")
ax.fill_between(u, level - 1.96 * level_std, level + 1.96 * level_std, alpha=0.4, label="95% interval")
ax.plot(u, level, label="smoothed level")
ax.plot(u, f, "k--", label="truth")
ax.legend()
ax.set_title("Smoothed level of the trend plus AR(1) model")
plt.show()

# %% [markdown]
# ## Validation against `statsmodels`
#
# The same model is `statsmodels`' unobserved components model with a smooth trend and an AR(1) component,
# whose state vector is also $(L_t, S_t, A_t)$.
# With the same prior on $X_0$, the Kalman filter and smoother of both libraries agree:

# %%
mod_sm = tsa.UnobservedComponents(y, "smooth trend", autoregressive=1)
mod_sm.ssm.initialize_known(μ_0, Γ_0)
res_sm = mod_sm.smooth(fit.params)

np.testing.assert_allclose(fit.m_t_given_t, res_sm.filtered_state.T, rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(fit.Q_t_given_t, res_sm.filtered_state_cov.transpose(2, 0, 1), rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(fit.m_t_given_T, res_sm.smoothed_state.T, rtol=1e-5, atol=1e-5)

# %% [markdown]
# The smoothed covariances agree too, except at the first three time steps:

# %%
Q_sm = res_sm.smoothed_state_cov.transpose(2, 0, 1)
np.testing.assert_allclose(fit.Q_t_given_T[3:], Q_sm[3:], rtol=1e-5, atol=1e-5)
pd.DataFrame(
    {"yassm": fit.Q_t_given_T[:4, 1, 1], "statsmodels": Q_sm[:4, 1, 1]},
    index=pd.Index(range(4), name="t"),
).rename_axis(columns="smoothed variance of the slope")

# %% [markdown]
# `statsmodels`' values there lose precision to the $10^8$ prior variance:
# the slope variance cannot drop by orders of magnitude in a few steps when its innovations have a variance of
# only `slope_cov` (about $2 \times 10^{-5}$), and the covariance matrices are not even symmetric:

# %%
pd.Series(
    {
        "yassm": np.abs(fit.Q_t_given_T[:3] - fit.Q_t_given_T[:3].transpose(0, 2, 1)).max(),
        "statsmodels": np.abs(Q_sm[:3] - Q_sm[:3].transpose(0, 2, 1)).max(),
    },
    name="largest asymmetry of the smoothed covariances at t < 3",
)
