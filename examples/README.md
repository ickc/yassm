# Examples

Tutorials on state space modelling with yassm, as [jupytext](https://jupytext.readthedocs.io/) percent-format scripts:
run them as plain Python scripts, or open them as notebooks in Jupyter.
They are rendered with their outputs in the [documentation](https://ickc.github.io/yassm/examples/).

| Example | Topics |
| --- | --- |
| [`01_kalman_filter.py`](01_kalman_filter.py) | Local level model of the Nile: Kalman filter and smoother, EM, forecasting, missing data |
| [`02_maximum_likelihood.py`](02_maximum_likelihood.py) | EM, gradient ascent and `scipy.optimize`; standard errors; AR(1) plus noise |
| [`03_custom_models.py`](03_custom_models.py) | Trend extraction; defining a model with `LinearGaussianModel`; validation against statsmodels |
| [`04_bayesian_inference.py`](04_bayesian_inference.py) | Simulation smoothing (FFBS) and Gibbs sampling, and a degenerate chain |
| [`05_arma_models.py`](05_arma_models.py) | Exact AR(2) posterior of sunspot numbers; the two maxima of an MA(1) likelihood |
| [`06_grid_filter.py`](06_grid_filter.py) | Exact filtering of a non-Gaussian model on a grid: level shifts |
| [`07_particle_filters.py`](07_particle_filters.py) | Bootstrap and optimal particle filters against the Kalman filter; likelihood by grid search |
| [`08_heavy_tailed_noise.py`](08_heavy_tailed_noise.py) | Cauchy noise with a non-linear particle filter; model comparison on the Nile |
| [`09_change_points.py`](09_change_points.py) | Level shifts, trend breaks and volatility regimes with jump processes |

The datasets and their sources are listed in [`data/README.md`](data/README.md).

Run one with

```sh
pixi run python examples/01_kalman_filter.py
```

or all of them, with reduced sample sizes, as integration tests:

```sh
pixi run examples
```

`YASSM_EXAMPLES_FAST=1` shrinks the sample sizes and grids so that each example runs in seconds;
the results are then not meaningful.
