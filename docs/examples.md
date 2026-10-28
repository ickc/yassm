# Examples

Each example is a [jupytext](https://jupytext.readthedocs.io/) script in
[`examples/`](https://github.com/ickc/yassm/tree/main/examples),
which runs as a plain Python script or opens as a notebook,
and is tested as part of the integration tests (`pixi run examples`).

```{toctree}
:caption: Linear Gaussian models
:maxdepth: 1

examples/01_kalman_filter
examples/02_maximum_likelihood
examples/03_custom_models
examples/04_bayesian_inference
examples/05_arma_models
```

```{toctree}
:caption: Beyond linear Gaussian models
:maxdepth: 1

examples/06_grid_filter
examples/07_particle_filters
examples/08_heavy_tailed_noise
examples/09_change_points
```
