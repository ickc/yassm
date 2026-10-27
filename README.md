# yassm — Yet Another State Space Model package

[![CI](https://github.com/ickc/yassm/actions/workflows/ci.yml/badge.svg)](https://github.com/ickc/yassm/actions/workflows/ci.yml)
[![Documentation](https://github.com/ickc/yassm/actions/workflows/docs.yml/badge.svg)](https://ickc.github.io/yassm)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

State space models for time series analysis, implemented from scratch in Python and [Numba](https://numba.pydata.org/).

I wrote yassm for STAT 248 *Analysis of Time Series* at UC Berkeley (spring 2022).
The course was taught in R; instead, I implemented every method in this package and used it for the homework,
which is included here as [examples](examples/stat248) and integration tests.

## Features

- **Linear Gaussian state space models**: Kalman filter, smoother and forecast;
  maximum likelihood by EM or `scipy.optimize`;
  forward-filtering backward-sampling and Gibbs sampling;
  ready-made local level, local linear trend, local ratio, AR(2) and MA(1) models.
- **Grid methods** for non-Gaussian one-dimensional models.
- **Particle filters**: bootstrap and optimal (locally adapted) proposals,
  general non-linear models, likelihood maximisation by grid search or `scipy.optimize`.
- **Quasi-Monte Carlo** building blocks for sequential quasi-Monte Carlo (Gerber & Chopin, 2015):
  Hilbert sort in any dimension without Hilbert indices, Halton sequences, Gaussian transformations.

All the numerical kernels are compiled ahead of time with Numba (`nopython`, `nogil`).

## Quick start

```python
import numpy as np
from yassm.linear_gaussian.model.local_level import local_level_model

rng = np.random.default_rng(0)
y = np.cumsum(rng.normal(size=200)) + rng.normal(scale=2.0, size=200)

# local level model with a diffuse prior; MLE of (σ_ε², σ_Z²) by EM
fit = local_level_model.fit(y, None, μ_0=0.0, Γ_0=1e8, mle_method="em", mle_n_iter=100)
print(fit.params, fit.log_likelihood)
fit.plot(T_forecast=20)  # smoothed state and forecast with confidence intervals
```

See the [documentation](https://ickc.github.io/yassm) for the API reference,
a [quasi-Monte Carlo demo](https://ickc.github.io/yassm/qmc/) and the rendered homework.

## Installation

yassm is not on PyPI (the name is taken by an unrelated project). Install it from GitHub:

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
pixi run examples   # homework scripts as integration tests
pixi run lint       # ruff
pixi run docs       # documentation into dist/docs
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more.
