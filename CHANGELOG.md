# Changelog

## Unreleased

### Changed

- Support Python 3.11–3.15, NumPy 2 and Numba ≥ 0.59 (`numba.generated_jit` was removed).
- Packaging: PEP 621 metadata built with hatchling; development environments and tasks managed by pixi.
- The core dependencies are now only NumPy, SciPy and Numba; matplotlib/seaborn, pandas and numdifftools are optional
  extras (`plot`, `pandas`, `hessian`, `all`). seaborn, used by `plot`, was previously undeclared.
- Importing yassm no longer configures logging; set `YASSMLOGLEVEL` or call `yassm.util.setup_logging()`.
- Docstrings use the NumPy style.
- `GeneralNonLinearBootstrapParticleFitlerStateSpaceModel` and `hilber_sort_inplace_func` are renamed to
  `GeneralNonLinearBootstrapParticleFilterStateSpaceModel` and `hilbert_sort_inplace_func`;
  the old names remain as aliases.

### Fixed

- Halton/van der Corput sequences truncated bases to `int8`, breaking `halton_sequence` for dimensions ≥ 32.
- The linear range reduction of the full-range Hilbert sort mapped the maximum of each axis to the middle of the range.
- The whitening Cholesky decomposition was made "positive definite" by shifting its spectrum the wrong way,
  and failed for singular covariances.
- Hilbert sort with order 0 scrambled the array; full-range Hilbert sort failed for a single point.
- `random_choice_weighted_inplace` clamped indices by the number of samples instead of the number of bins,
  reading out of bounds when drawing more samples than bins.
- Read-only input arrays (e.g. `DataFrame.values` under pandas' Copy-on-Write) were rejected.
- `mle_method="scipy_minimize"` of the particle filters failed with recent SciPy passing non-contiguous arrays.

### Added

- The STAT 248 homework solutions as examples and integration tests.
- Documentation site with an API reference, a quasi-Monte Carlo demo and the examples.

## 0.1.0

- First release and proof of concept.
