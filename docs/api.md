# API reference

## Top level

The main classes and ready-made models can be imported from `yassm` directly, e.g. `from yassm import local_level_model`.

```{eval-rst}
.. autosummary::

   ~yassm.linear_gaussian.LinearGaussianModel
   ~yassm.linear_gaussian.LinearGaussianFit
   ~yassm.linear_gaussian.model.local_level.local_level_model
   ~yassm.linear_gaussian.model.local_linear.local_linear_model
   ~yassm.linear_gaussian.model.local_ratio.local_ratio_model
   ~yassm.linear_gaussian.model.moving_average_1.moving_average_1_model
   ~yassm.grid.one_dimension.GridFilterModel
   ~yassm.grid.one_dimension.GridFilterFit
   ~yassm.particle_filter.LinearGaussianParticleFilterModel
   ~yassm.particle_filter.NonLinearParticleFilterModel
   ~yassm.particle_filter.ParticleFilterFit
   ~yassm.qmc.random.HaltonSequence
   ~yassm.qmc.hilbert_curve.HilbertCurveInt
   ~yassm.qmc.hilbert_curve.HilbertCurveFloat01
   ~yassm.qmc.hilbert_curve.HilbertCurveFloat
```

## Array shapes

Arrays are annotated with their dtype and shape using [jaxtyping](https://docs.kidger.site/jaxtyping/),
e.g. `Float64[ndarray, 'T d']` is a float64 array of shape `(T, d)`. The dimensions are named:

| Name | Dimension |
| --- | --- |
| `T` | time steps |
| `n` | observations |
| `d` | states |
| `p` | parameters |
| `N` | draws: samples, particles, or iterations of a chain |
| `K` | parameter sets, e.g. a grid of parameters |
| `G` | grid points of a grid filter |
| `T_forecast` | forecast time steps |

The test suite checks these annotations at runtime with [beartype](https://beartype.readthedocs.io).
To check them in your own code too, set the environment variable `YASSM_TYPECHECK=1` before importing yassm,
with beartype installed.

## Modules

```{eval-rst}
.. autosummary::
   :toctree: _generated/api
   :recursive:

   yassm
```
