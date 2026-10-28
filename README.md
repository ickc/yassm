# yassm — Yet Another State Space Model package

[![CI](https://github.com/ickc/yassm/actions/workflows/ci.yml/badge.svg)](https://github.com/ickc/yassm/actions/workflows/ci.yml)
[![Documentation](https://github.com/ickc/yassm/actions/workflows/docs.yml/badge.svg)](https://ickc.github.io/yassm)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

State space models for time series analysis in Python, with every algorithm compiled by
[Numba](https://numba.pydata.org/): Kalman filters and smoothers, maximum likelihood and Bayesian inference,
grid filters for non-Gaussian models, particle filters for non-linear ones,
and quasi-Monte Carlo building blocks.

## Features

- **Linear Gaussian models**: Kalman filter, smoother and forecasts, with missing observations;
  maximum likelihood by EM, gradient ascent or `scipy.optimize`, with standard errors;
  simulation smoothing (forward-filtering backward-sampling) and Gibbs sampling.
  Ready-made local level, local linear trend, AR(1) plus noise and MA(1) models,
  or your own model from a few Numba functions of its parameters.
- **Grid filters**: exact filtering, smoothing and likelihood for any one-dimensional model on a discretised state.
- **Particle filters**: bootstrap and optimal (locally adapted) proposals for linear Gaussian models,
  and bootstrap filters for arbitrary non-linear, non-Gaussian models, with likelihood estimation.
- **Quasi-Monte Carlo**: Hilbert sort in any dimension without Hilbert indices, Halton sequences,
  and their transformation to correlated Gaussians, the building blocks of sequential quasi-Monte Carlo
  (Gerber & Chopin, 2015).

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

The [documentation](https://ickc.github.io/yassm) has the API reference and worked
[examples](https://ickc.github.io/yassm/examples/), from Kalman filtering of the Nile to change-point detection
with particle filters, each validated against `statsmodels` or exact results where possible.

## Installation

yassm is not on PyPI yet. Install it from GitHub:

```sh
uv pip install "yassm[all] @ git+https://github.com/ickc/yassm.git"
```

The core dependencies are NumPy, SciPy and Numba; the extras are `plot` (matplotlib, seaborn),
`pandas`, `hessian` (numdifftools) and `all`.

## Development

Environments and tasks are managed by [pixi](https://pixi.sh), with conda-forge packages
on Linux (x86-64, aarch64), macOS (x86-64, arm64) and Windows:

```sh
pixi run test       # unit tests
pixi run examples   # the examples as integration tests
pixi run lint       # ruff
pixi run docs       # documentation into dist/docs
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more.

## History

yassm started in 2022 as a from-scratch implementation of the methods of a time series course at UC Berkeley,
which was taught in R.
