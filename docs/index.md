# yassm

*Yet Another State Space Model package*: state space models for time series analysis,
implemented from scratch in Python and [Numba](https://numba.pydata.org/).

yassm was written for the STAT 248 *Analysis of Time Series* course at UC Berkeley (spring 2022),
which was taught in R, as a reimplementation of its methods in Python:

- **Linear Gaussian state space models**: Kalman filter, smoother and forecast,
  maximum likelihood by EM or `scipy.optimize`, forward-filtering backward-sampling and Gibbs sampling,
  with ready-made local level, local linear trend, local ratio, AR(2) and MA(1) models;
- **Grid methods** for non-Gaussian one-dimensional models;
- **Particle filters**: bootstrap and optimal (locally adapted) proposals,
  including non-linear models, with likelihood maximisation by grid search or `scipy.optimize`;
- **Quasi-Monte Carlo** building blocks for sequential quasi-Monte Carlo:
  Hilbert sort in any dimension, Halton sequences and Gaussian transformations.

```{toctree}
:maxdepth: 2

installation
qmc
examples
api
contributing
changelog
```
