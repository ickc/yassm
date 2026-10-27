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
# # HW 4: Problem 4
#
# *Originally written on 2022-04-08.*

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from stat248_data import load_SN_y_tot_V2

# %%
from yassm.lib.linear_gaussian.model.autoregressive_2 import direct_sampling, neg_log_likelihood

# %% [markdown]
# See <https://wwwbis.sidc.be/silso/infosnytot>,
#
#     Column 1: Gregorian calendar year (mid-year date)
#     Column 2: Yearly mean total sunspot number.
#     Column 3: Yearly mean standard deviation of the input sunspot numbers from individual stations.
#     Column 4: Number of observations used to compute the yearly mean total sunspot number.
#     Column 5: Definitive/provisional marker. A blank indicates that the value is definitive. A '*' symbol indicates that the yearly average still contains provisional daily values and is subject to a possible revision.
# %%
df = load_SN_y_tot_V2()

# %%
# copy: Numba-jitted functions reject the read-only arrays returned by pandas (Copy-on-Write)
y = df.SNvalue.to_numpy(copy=True)
t = df.index.values

# %%
df["SNvalue"].plot()

# %% [markdown]
# ## Part d
#
# The function for the sampling is in
#
# -   `yassm.lib.linear_gaussian.model.autoregressive_2.direct_sampling`.
#
# See documentation attached at the end for details.
#
# Checking the MLE first,

# %%
# %%time
res = scipy.optimize.minimize(
    neg_log_likelihood,
    [y.std(), 0.0, 1.0, 0.0],
    args=(y,),
    bounds=(
        (0.0, None),
        (None, None),
        (None, None),
        (None, None),
    ),
)
res

# %%
# %%time
params_chain = direct_sampling(y, 400)

# %% [markdown]
# ## Part e

# %%
mle = np.append(np.sqrt(res.x[0]), res.x)

# %%
# first column is σ_Z square
params_chain_with_sqrt = np.append(np.sqrt(params_chain[:, 0]).reshape(-1, 1), params_chain, axis=1)

# %%
names = ["σ_Z", "σ_Z-square", "φ_0", "φ_1", "φ_2"]

# %%
df_q4 = pd.DataFrame(
    {
        "MLE": mle,
        "mean": params_chain_with_sqrt.mean(0),
        "std": params_chain_with_sqrt.std(0),
    },
    index=names,
)
df_q4

# %%
fig, axes = plt.subplots(len(names))
for i, ax in enumerate(axes):
    sns.histplot(params_chain_with_sqrt[:, i], kde=True, ax=ax)
    ax.set_title(names[i])

# %% [markdown]
# Now we compare ours to `statsmodels`,

# %%
from statsmodels.tsa.arima.model import ARIMA

# %%
arma_mod20 = ARIMA(y, order=(2, 0, 0)).fit()

# %% [markdown]
# We can't reproduce the log-likelihood calculated via `statsmodels`,
# so perhaps they used a different one or has a different proportionality constants in likelihood.

# %%
params_sm = np.empty(4)
params_sm[0] = arma_mod20.params[3]
params_sm[1:] = arma_mod20.params[:3]
-neg_log_likelihood(params_sm, y)

# %%
arma_mod20.summary()

# %% [markdown]
# Comparing that to ours,

# %%
df_q4

# %% [markdown]
# We see that all the parameters are in agreement w.r.t. to standard errors
# except for the constant term $φ_0$.
#
# In particular for $σ_Z$ that the question called for,
# since `statsmodels` returns its square,
# we should be comparing $σ_Z^2$ instead.
#
# It is interesting that the $\hat{σ}_Z^2$ provided by `statsmodels`
# actually is in the middle of our MLE $\hat{σ}_Z^2$ and mean $\hat{σ}_Z^2$.
