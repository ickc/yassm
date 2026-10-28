# Contributing

Bug reports and pull requests are welcome at <https://github.com/ickc/yassm/issues>.
When reporting a bug, please include your OS, the output of `pixi list` (or `pip list`), and steps to reproduce it.

## Development

The development environments are managed by [pixi](https://pixi.sh), with every package from conda-forge
and yassm itself installed in editable mode:

```sh
git clone https://github.com/ickc/yassm.git
cd yassm
pixi run test              # unit tests
pixi run test-boundscheck  # unit tests with Numba bounds checking, to catch out-of-bounds accesses
pixi run examples          # the examples as integration tests
pixi run lint              # ruff lint and format check
pixi run fmt               # ruff auto-fix and format
pixi run typecheck         # mypy
pixi run docs              # build the documentation into dist/docs
pixi run -e py311 test     # tests on another Python version (py311 ... py315)
```

Without pixi, `uv sync --group dev` (or `pip install -e . --group dev`) installs the same tools from PyPI.

## Conventions

- Code is formatted and linted with [ruff](https://docs.astral.sh/ruff/) (`pixi run fmt`).
- The package is type checked with [mypy](https://mypy.readthedocs.io) (`pixi run typecheck`).
- Docstrings use the [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html).
- Numba functions are compiled eagerly with explicit signatures, `nopython=True, nogil=True`;
  use `cache=True` unless the function is a closure, uses dynamic globals, or is self-recursive
  (a cached self-recursive function crashes when reloaded from the cache).
- Every bug fix comes with a regression test.
