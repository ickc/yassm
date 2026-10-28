# yassm

*Yet Another State Space Model package*: state space models for time series analysis in Python,
with every algorithm compiled by [Numba](https://numba.pydata.org/).

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Linear Gaussian models
Kalman filter, smoother and forecasts; maximum likelihood by EM, gradient ascent or `scipy.optimize`;
simulation smoothing and Gibbs sampling. Ready-made local level, local linear trend, AR(1) plus noise and MA(1)
models, or your own from a few Numba functions.
:::

:::{grid-item-card} Grid filters
Exact filtering and smoothing of any one-dimensional model, Gaussian or not,
on a discretised state space.
:::

:::{grid-item-card} Particle filters
Bootstrap and optimal (locally adapted) proposals for linear Gaussian models,
and bootstrap particle filters for arbitrary non-linear, non-Gaussian models, with likelihood estimation.
:::

:::{grid-item-card} Quasi-Monte Carlo
Building blocks of sequential quasi-Monte Carlo: Hilbert sort in any dimension, Halton sequences,
and their transformation to correlated Gaussians.
:::
::::

## Quick start

```python
import numpy as np
from yassm import local_level_model

rng = np.random.default_rng(0)
y = np.cumsum(rng.normal(size=200)) + rng.normal(scale=2.0, size=200)

# local level model with a diffuse prior; MLE of (σ_ε², σ_Z²) by EM
fit = local_level_model.fit(y, None, μ_0=0.0, Γ_0=np.inf, mle_method="em", mle_n_iter=100)
print(fit.params_to_series)
fit.plot(T_forecast=20)  # smoothed state and forecast with 95% intervals
```

Start with {doc}`examples/01_kalman_filter`, or browse the {doc}`examples`.

```{toctree}
:hidden:

installation
examples
qmc
api
contributing
changelog
```

## History

yassm started in 2022 as a from-scratch implementation of the methods of a time series course at UC Berkeley,
which was taught in R.
