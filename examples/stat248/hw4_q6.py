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
# # HW 4: Problem 6
#
# *Originally written on 2022-04-10.*

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit
from stat248_data import load_Prob6HW4DataStat248, size

from yassm.lib.linear_gaussian import prob_normal
from yassm.linear_gaussian.model.moving_average_1 import moving_average_1_model

if TYPE_CHECKING:
    import numpy.typing as npt

# %%
df = load_Prob6HW4DataStat248()

# %%
df

# %%
df.plot()

# %%
# copy: Numba-jitted functions reject the read-only arrays returned by pandas (Copy-on-Write)
y = df.y.to_numpy(copy=True)
t = df.index.values

# %% [markdown]
# ## Part b
#
# Main module implementing this model is in
# `yassm.lib.linear_gaussian.model.moving_average_1`.
# See documentation attached at the end for details.
#
# $μ_0, Γ_0$ is still passed here as it was the original assumption in structuring `yassm`,
# but it is overridden by
#
#     yassm.lib.linear_gaussian.model.moving_average_1.X0
#
# which is implemented in
#
# -   `yassm.linear_gaussian.GeneralLinearGaussianStateSpaceModel.kalman_filtering_inplace`,
# -   `yassm.linear_gaussian.GeneralLinearGaussianStateSpaceModel.neg2_log_likelihood`.

# %%
μ_0_fake = np.zeros(2)
Γ_0_fake = np.identity(2)

# %%
# %%time
fit = moving_average_1_model.fit(y, [y.std(), 0.0], μ_0_fake, Γ_0_fake, mle_n_iter=1, mle_return_chain=True, t=t)
fit.params, fit.log_likelihood

# %%
fit.mle_params_chain_to_frame.plot(title="Chain towards MLE")

# %% [markdown]
# There is another local maximum,

# %%
# %%time
fit2 = moving_average_1_model.fit(y, [y.std(), -2.0], μ_0_fake, Γ_0_fake, mle_n_iter=1, mle_return_chain=True, t=t)
fit2.params, fit2.log_likelihood

# %%
fit2.mle_params_chain_to_frame.plot(title="Chain towards MLE")

# %% [markdown]
# Note that since we calculate the likelihood with relative constant,
# we normalize the peak to 1 here.

# %%
# %%time
neg2_log_likelihoods = moving_average_1_model.neg2_log_likelihoods


@jit("f8[::1](f8[:, ::1], f8[::1], f8)", nopython=True, nogil=True)  # , parallel=True)
def posterior(
    paramses: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mle_log_likelihood: float,
):
    """Posterior of parameters given Ys.

    :param paramses: 2d-array where each row is a params.
        σ_sq, θ = params
    :param mle_log_likelihood: log-likelihood at MLE, used to normalize
        peak to 1. set this to 0.0 to disable.
    """
    likelihoods = np.exp(
        -0.5
        * neg2_log_likelihoods(
            paramses,
            y.reshape(-1, 1),
            # fake values, will be overriden inside neg2_log_likelihoods
            np.zeros(2),
            np.identity(2),
        )
        - mle_log_likelihood
    )
    σ_sqs = paramses[:, 0]
    # prior of log(σ) is uniform
    return likelihoods / np.sqrt(σ_sqs)


# %% [markdown]
# Plot around MLE,

# %%
# range from ratio to value
r_σ = 0.99
r_θ = 1.0

σ_mle_sq, θ_mle = fit.params
σ_mle = np.sqrt(σ_mle_sq)
σ_low = (1.0 - r_σ) * σ_mle
σ_high = (1.0 + r_σ) * σ_mle
θ_low = (1.0 - r_θ) * θ_mle
θ_high = (1.0 + r_θ) * θ_mle
(σ_low, σ_mle, σ_high), (θ_low, θ_mle, θ_high)

# %%
n = size(1000, fast=10)
σs, θs = np.meshgrid(
    np.linspace(σ_low, σ_high, n),
    np.linspace(θ_low, θ_high, n),
)

# %%
params = np.column_stack((np.square(σs.flatten()), θs.flatten()))

# %%
# %%time
p = posterior(params, y, fit.log_likelihood)

# %%
p_2d = p.reshape((n, n))

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(σs, θs, p_2d)
ax.set_xlabel("σ")
ax.set_ylabel("θ")

# %% [markdown]
# ## Part c
#
# Main function calculating the hessian is
#
# -   `yassm.linear_gaussian.GeneralLinearGaussianStateSpaceModelFit.mle_hessian_log_likelihood`.

# %%
Σ_inv = fit.posterior_normal_approximation_cov_inv

# %% [markdown]
# Again we normalize the peak to 1.

# %%
p_normal = prob_normal(params, fit.params, Σ_inv, True)
p_normal_2d = p_normal.reshape((n, n))

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(σs, θs, p_normal_2d)
ax.set_xlabel("σ")
ax.set_ylabel("θ")

# %% [markdown]
# Repeat to the other local maximum:

# %%
Σ_inv2 = fit2.posterior_normal_approximation_cov_inv

# %% [markdown]
# Again we normalize the peak to 1.

# %%
p_normal2 = prob_normal(params, fit2.params, Σ_inv2, True)
p_normal_2d2 = p_normal2.reshape((n, n))

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(σs, θs, p_normal_2d2)
ax.set_xlabel("σ")
ax.set_ylabel("θ")

# %% [markdown]
# Overlapping the two, per peak, we can hardly see any differences.
# But neither of the approximation fit the actual likelihood everywhere.

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(σs, θs, p_2d)
ax.plot_surface(σs, θs, p_normal_2d)
ax.plot_surface(σs, θs, p_normal_2d2)
ax.set_xlabel("σ")
ax.set_ylabel("θ")

# %% [markdown]
# ## Part d

# %%
Σ = fit.posterior_normal_approximation_cov

# %%
Σ

# %% [markdown]
# Off-diagonal are negligible, treating them as independent and estimate individual $σ$.

# %%
σ_mle = np.sqrt(Σ[np.diag_indices(2)])

# %%
pd.DataFrame([fit.params, σ_mle], columns=fit.param_names, index=["MLE", "σ"])

# %%
from statsmodels.tsa.arima.model import ARIMA

# %%
mod_sm = ARIMA(y, order=(0, 0, 1))

# %%
res = mod_sm.fit()

# %%
res.summary()

# %% [markdown]
# Comparing to the result from `statsmodels` (look for `ma.L1` in the above table), the two are pretty similar.
#
# Note that `statsmodels`'s model has an extra constant term which isn't significant, so effectively our assumption of it being $0$ gives almost identical results.

# %%
Σ2 = fit2.posterior_normal_approximation_cov

# %%
Σ2

# %% [markdown]
# Off-diagonal are significant for the other peak, treating them as independent and estimate individual $σ$ anyway.

# %%
σ_mle2 = np.sqrt(Σ2[np.diag_indices(2)])

# %%
pd.DataFrame([fit2.params, σ_mle2], columns=fit.param_names, index=["MLE", "σ"])

# %% [markdown]
# Note that the `statsmodels` did not cover this other maximum.
