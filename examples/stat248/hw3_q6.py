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

# %% [markdown] tags=[]
# # HW 3: Problem 6
#
# *Originally written on 2022-03-16.*

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit
from stat248_data import load_Prob6HW3DataStat248, size

if TYPE_CHECKING:
    import numpy.typing as npt

# %%
df = load_Prob6HW3DataStat248()
y = df.y.values
t = np.linspace(0.0, 1.0, y.size)

# %%
df

# %%
df.plot()

# %% [markdown]
# ## Part a

# %%
from yassm.grid.one_dimension import General1DStateSpaceModel
from yassm.lib.grid.one_dimension import prob_normal


# %%
@jit("f8(f8, f8, f8[::1], f8)", nopython=True, nogil=True, cache=True)
def transition_func(
    x_t: float,
    x_tMinus1: float,
    params: npt.NDArray[np.float64],
    Δx: float,
) -> float:
    """
    X_t = A_t X_(t-1) + U_t
    U_t ~ N(0, Σ_t)
    """
    σ_Z_sq, α = params
    # σ_ε_sq = 25.
    return (1.0 - α) * prob_normal(x_t, x_tMinus1, 0.00001, Δx) + α * prob_normal(x_t, x_tMinus1, σ_Z_sq, Δx)


@jit("f8(f8, f8, f8[::1], f8)", nopython=True, nogil=True, cache=True)
def design_func(
    y_t: float,
    x_t: float,
    params: npt.NDArray[np.float64],
    Δx: float,
) -> float:
    """
    Y_t = B_t X_t + V_t
    V_t ~ N(0, R_t)
    """
    # σ_Z_sq, α = params
    σ_ε_sq = 25.0
    return prob_normal(y_t, x_t, σ_ε_sq, Δx)


# %%
G = size(100, fast=20)

y_min = y.min()
y_max = y.max()
y_range = 0.5 * (y_max - y_min)
y_mid = 0.5 * (y_min + y_max)
# how many times range we want to go out
# if n_range = 1, then from y_min to y_min
n_range = 1.5
x_0 = y_mid - n_range * y_range
x_f = y_mid + n_range * y_range
y_min, y_max, x_0, x_f

# %%
mod = General1DStateSpaceModel(
    2,
    transition_func,
    design_func,
    x_0,
    x_f,
    G,
    param_names=["σ_Z_sq", "α"],
)

# %%
# %%time
res = mod.fit(
    y,
    np.array([0.1, 0.01]),
    np.full(G, 1.0 / G),
    mle_kwargs={"bounds": ((0.0, None), (0.0, 1.0))},
    mle_n_iter=1,
    mle_return_chain=True,
)

# %%
σ_Z_sq, α_hat = res.params
σ_Z = np.sqrt(σ_Z_sq)


# %%
pd.Series({"σ̂_Z": σ_Z, "α̂": α_hat}, name="MLE")

# %% [markdown]
# ## Part b

# %%
fig, ax = plt.subplots()
df.plot(ax=ax)
ax.plot(res.t_array, res.m_t_given_T, label="grid-method-filtered")
