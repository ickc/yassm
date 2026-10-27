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
# # HW 3: Problem 1
#
# *Originally written on 2022-03-11.*

# %%
import numpy as np

# %%
from stat248_data import load_nile

# %% [markdown]
# I wrote the `yassm` model for this class. Documentation of `yassm` attached as an appendix.
#
# TODO: made sure I include `yassm` documentation.
# %%
from yassm.linear_gaussian.model.local_level import local_level_model

# %%
df = load_nile()

# %%
df

# %%
y = df.Nile.values
t = df.index.values

# %%
μ_0 = 0.0
Γ_0 = np.inf

# %%
# law of large number guesstimate
# ~ 2 * σ_ε_sq + 2 * σ_Z_sq
temp1 = np.nanvar(y[2:] - y[:-2])
# ~ 2 * σ_ε_sq + σ_Z_sq
temp2 = np.nanvar(np.diff(y))
σ_Z_sq0 = temp1 - temp2
σ_ε_sq0 = 0.5 * (temp2 - σ_Z_sq0)
params0 = [σ_ε_sq0, σ_Z_sq0]
params0

# %%
# %%time
res = local_level_model.fit(y, params0, μ_0, Γ_0, mle_n_iter=1000, mle_return_chain=True, t=t)

# %% [markdown]
# This shows the iterations by the EM algorithm,

# %%
res.mle_params_chain_to_frame.plot(logx=True, logy=True, title="Iterations by the EM algorithm")

# %% [markdown]
# Answer:

# %%
res.params_to_series.to_frame("parameters")

# %%
res.plot(10)

# %% [markdown]
# Now let's compare to `statsmodels`,

# %%
import statsmodels.tsa.api as tsa

# %%
# %%time
mod = tsa.UnobservedComponents(df, "local level")
res_sm = mod.fit()

# %% [markdown]
# Comparing to the above,

# %%
res_sm.params.to_frame("parameters")

# %%
fig = res_sm.plot_components(
    which="smoothed",
)

# %% [markdown]
# Comparison:
#
# The parameters are close,
# but $σ_Z$ (`state_cov`) is larger and $σ_ε$ (`obs_cov`) is smaller from `statsmodels`.
#
# Looking at the plot "Iterations by the EM algorithm",
# it may means that the result from `statsmodels` terminates earlier and is less optimal.
