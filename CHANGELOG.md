# Changelog

## Unreleased

### Changed

- Support Python 3.11–3.15, NumPy 2 and Numba ≥ 0.59 (`numba.generated_jit` was removed).
- Packaging: PEP 621 metadata built with hatchling; development environments and tasks managed by pixi.
- The core dependencies are now only NumPy, SciPy and Numba; matplotlib/seaborn, pandas and numdifftools are optional
  extras (`plot`, `pandas`, `hessian`, `all`). seaborn, used by `plot`, was previously undeclared.
- Importing yassm no longer configures logging: its warnings are still printed, but set `YASSMLOGLEVEL` or call
  `yassm.util.setup_logging()` to see its progress messages.
- Docstrings use the NumPy style.
- Shorter class names; the old names still work but emit a `DeprecationWarning`:

  | Old | New |
  | --- | --- |
  | `GeneralLinearGaussianStateSpaceModel` | `LinearGaussianModel` |
  | `GeneralLinearGaussianStateSpaceModelFit` | `LinearGaussianFit` |
  | `GenericLinearGaussianStateSpaceModel` | `BaseLinearGaussianModel` |
  | `General1DStateSpaceModel` | `GridFilterModel` |
  | `General1DStateSpaceModelFit` | `GridFilterFit` |
  | `GeneralLinearGaussianParticleFilterStateSpaceModel` | `LinearGaussianParticleFilterModel` |
  | `GeneralNonLinearBootstrapParticleFilterStateSpaceModel` (and its misspelling `...Fitler...`) | `NonLinearParticleFilterModel` |
  | `GenericParticleFilterStateSpaceModel` | `BaseParticleFilterModel` |
  | `GenericParticleFilterStateSpaceModelFit` | `ParticleFilterFit` |
  | `hilber_sort_inplace_func` | `hilbert_sort_inplace_func` |

- The built-in models bound variances by machine epsilon instead of 0 in `scipy.optimize`.

### Added

- The main classes and ready-made models are exported from the top level, e.g. `from yassm import local_level_model`,
  and the package ships type information (`py.typed`), checked with mypy.
- `LinearGaussianFit.run_gibbs_sampling(filter_n_std=...)` re-runs the Gibbs sampler,
  optionally keeping only the draws near the MLE.
- `GridFilterFit.params_to_series`, like the other fits.
- Examples: nine tutorials, from Kalman filtering to change-point detection with particle filters,
  run as integration tests and rendered in the documentation.
- Documentation site with an API reference, the examples and a quasi-Monte Carlo demo.

### Fixed

- The optimal particle filter omitted $\log p(y_1)$ from its log-likelihood (replacing it by $-\log N$),
  biasing likelihood estimates and maximum likelihood.
- The AR(2) log-likelihood counted $n + 3$ instead of $n$ residuals, biasing the MLE of $σ^2$.
- The Kalman log-likelihood was NaN as soon as an observation was missing, so that `scipy_minimize` could not
  estimate the MLE of such data: missing observations now have no term, and the likelihood is conditional on the
  first observation that is not missing.
- With $n$-dimensional observations, the log-likelihood of `LinearGaussianModel` and the weights of the particle
  filters were normalised as for $n = 1$: log-likelihoods were too high by $(n - 1) \log(2π) / 2$ per time step.
- `mle_hessian_log_likelihood` and `posterior_normal_approximation_cov` raised a `TypeError` for the built-in local
  level and local ratio models, and returned NaN for small variances.
- With a variance bound of exactly 0, `scipy.optimize` could stop early at a singular point while reporting success,
  e.g. for the MA(1) model.
- `fit(params=None, mle_method="grid_search")` of the particle filters required a `params_guess`.
- With `mle_n_iter` but no or an unknown `mle_method`, the particle filters silently skipped the MLE: no method
  now means `scipy_minimize`, and an unknown one raises a `ValueError`.
- `gibbs_params_chain_with_burn_in` and `gibbs_X_chain_with_burn_in` did not exist until another Gibbs result
  was accessed.
- `mle_n_iter` was reported as ignored by `scipy_minimize` and `grid_search` even though a positive value is required.
- `HilbertCurve*.plot` drew 3-D curves on a 3-D axes stacked over a stray 2-D one, and ignored a given 3-D axes;
  `plot` of the fits and of 2-D curves used the current figure instead of the one of the given axes.
- Halton/van der Corput sequences truncated bases to `int8`, breaking `halton_sequence` for dimensions ≥ 32.
- The linear range reduction of the full-range Hilbert sort mapped the maximum of each axis to the middle of the range.
- Its logistic and whitening range reductions sorted the points far in the upper tail first instead of last.
- The whitening Cholesky decomposition was made "positive definite" by shifting its spectrum the wrong way,
  and failed for singular covariances.
- The ZCA whitening range reductions failed for singular covariances, for which `scipy.linalg.sqrtm` can return
  a complex matrix, and `whitening_transform` failed for one-dimensional data.
- Hilbert sort with order 0 scrambled the array; full-range Hilbert sort failed for a single point.
- Hilbert sort left the last point of a group on the wrong side of a split whenever the rest of the group belonged
  to the first side, so that point sets other than full grids were not sorted along the curve.
- `random_choice_weighted_inplace` clamped indices by the number of samples instead of the number of bins,
  reading out of bounds when drawing more samples than bins.
- Read-only input arrays (e.g. `DataFrame.values` under pandas' Copy-on-Write) were rejected.
- `mle_method="scipy_minimize"` failed with recent SciPy passing non-contiguous arrays,
  and raised instead of backing off when the optimiser tried singular parameters.

### Performance

- Univariate models solve their Kalman updates without LAPACK, whose calls serialise across threads:
  `neg2_log_likelihoods` is about 5 times faster on 32 threads.

## 0.1.0

- First release and proof of concept.
