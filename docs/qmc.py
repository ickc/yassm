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
# # Quasi-Monte Carlo and Hilbert curves
#
# Sequential quasi-Monte Carlo (SQMC, [Gerber & Chopin 2015](https://doi.org/10.1111/rssb.12104))
# replaces the i.i.d. uniforms of a particle filter by a low-discrepancy point set,
# and orders the particles along a Hilbert curve
# so that the resampling step, which is one-dimensional, preserves their locality.
# `yassm.qmc` implements the two building blocks:
#
# - Hilbert sort of points in any dimension $d < 64$, without computing Hilbert indices
#   ([Imamura et al. 2016](https://doi.org/10.1007/978-3-319-46759-7_20)),
#   for points in $[0, 1)^d$ or for arbitrary real points after a range reduction;
# - Halton sequences and their transformation to correlated Gaussians.
#
# Everything heavy is compiled with Numba.

# %%
import matplotlib.pyplot as plt
import numpy as np

from yassm.qmc.hilbert_curve import HilbertCurveFloat, HilbertCurveInt
from yassm.qmc.random import HaltonSequence

# %% [markdown]
# ## Hilbert curves
#
# Sorting all the points of a $2^m \times 2^m$ grid traces the Hilbert curve of order $m$.

# %%
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
for order, ax in enumerate(axes, start=1):
    HilbertCurveInt.from_order(order, 2).to_float().plot(ax=ax, lw=1)
    ax.set_title(f"order {order}")
    ax.set_axis_off()

# %% [markdown]
# The same algorithm works in any dimension, here in 3D:

# %%
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection="3d")
array = HilbertCurveInt.from_order(3, 3).array
ax.plot(array[:, 0], array[:, 1], array[:, 2], lw=1)
ax.set_axis_off()

# %% [markdown]
# ## Hilbert-sorted Halton sequence
#
# A Halton sequence fills the unit square much more evenly than i.i.d. uniforms.
# Sorting it along the Hilbert curve gives a path through the points
# where consecutive points are close.

# %%
N = 512
halton = HaltonSequence(N, 2)
rng = np.random.default_rng(0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(*rng.uniform(size=(N, 2)).T, s=4)
axes[0].set_title("i.i.d. uniform")
axes[1].scatter(*halton.array.T, s=4)
axes[1].set_title("Halton")
h = halton.to_hilbert_curve()
h.sort()
h.plot(ax=axes[2], lw=0.8, marker=".", ms=3)
axes[2].set_title("Halton, Hilbert-sorted")
for ax in axes:
    ax.set_aspect("equal")

# %% [markdown]
# ## Correlated Gaussian quasi-random numbers
#
# `HaltonSequence.to_gaussian` Hilbert-sorts the Halton points in $[0, 1)^d$
# and maps them to $N(0, \Sigma)$.
# The result can be re-sorted in the real space,
# after mapping it back to a bounded box with a *range reduction*:
# a linear rescaling, the logistic map of Gerber & Chopin (2015),
# or a whitening transformation followed by the Gaussian CDF.

# %%
cov = np.array([[4.0, 1.5], [1.5, 1.0]])
gaussian = halton.to_gaussian(cov, method="ZCA_cor")
print("sample covariance:\n", np.cov(gaussian.array, rowvar=False))

methods = ["linear", "logistic", "whiten_ZCA_cor"]
fig, axes = plt.subplots(1, len(methods), figsize=(15, 5), sharex=True, sharey=True)
for method, ax in zip(methods, axes, strict=True):
    h = HilbertCurveFloat(gaussian.array.copy(), range_reduction_method=method, check_range=False)
    h.sort()
    h.plot(ax=ax, lw=0.6, marker=".", ms=3)
    ax.set_title(f"range reduction: {method}")
