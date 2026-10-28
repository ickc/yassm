# Installation

yassm is not published on PyPI yet. Install it from GitHub.

## With pixi (recommended for development)

[pixi](https://pixi.sh) sets up a conda-forge environment with everything needed to run the tests, examples and docs:

```sh
git clone https://github.com/ickc/yassm.git
cd yassm
pixi run test        # unit tests
pixi run examples    # run the examples as integration tests
pixi run docs        # build this documentation into dist/docs
pixi task list       # everything else
```

## With uv or pip

```sh
uv pip install "yassm[all] @ git+https://github.com/ickc/yassm.git"
# or
pip install "yassm[all] @ git+https://github.com/ickc/yassm.git"
```

The core dependencies are only NumPy, SciPy and Numba. The optional extras are

`plot`
: matplotlib and seaborn, for the `plot` methods;

`pandas`
: DataFrame helpers and datetime indices for forecasts;

`hessian`
: numdifftools, for the Hessian at the MLE (normal approximation of the posterior);

`all`
: all of the above.
