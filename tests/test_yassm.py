from __future__ import annotations

import importlib

import numpy as np
import pytest

from yassm.lib import sqrt_diag


def test_sqrt_diag() -> None:
    array = np.arange(12, dtype=np.float64).reshape((3, 2, 2))
    res = sqrt_diag(array)
    ref = np.sqrt(
        np.array(
            [
                [0, 3],
                [4, 7],
                [8, 11],
            ]
        )
    )
    np.testing.assert_array_equal(res, ref)


def test_import_does_not_require_optional_dependencies() -> None:
    """Importing every module must only need the core dependencies (numba, numpy, scipy)."""
    import subprocess
    import sys

    code = """
import pkgutil, importlib, sys
import yassm
for m in pkgutil.walk_packages(yassm.__path__, "yassm."):
    importlib.import_module(m.name)
optional = ("pandas", "matplotlib", "seaborn", "numdifftools", "statsmodels")
print(",".join(m for m in optional if m in sys.modules))
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert res.stdout.strip() == ""


def test_import_is_quiet() -> None:
    """A library should not configure logging on import."""
    import logging

    import yassm  # noqa: F401

    assert all(isinstance(h, logging.NullHandler) for h in logging.getLogger("yassm").handlers)


@pytest.mark.parametrize(
    ("module", "old", "new"),
    [
        ("yassm.linear_gaussian", "GeneralLinearGaussianStateSpaceModel", "LinearGaussianModel"),
        ("yassm.linear_gaussian", "GeneralLinearGaussianStateSpaceModelFit", "LinearGaussianFit"),
        ("yassm.linear_gaussian", "GenericLinearGaussianStateSpaceModel", "BaseLinearGaussianModel"),
        ("yassm.grid.one_dimension", "General1DStateSpaceModel", "GridFilterModel"),
        ("yassm.grid.one_dimension", "General1DStateSpaceModelFit", "GridFilterFit"),
        ("yassm.particle_filter", "GenericParticleFilterStateSpaceModel", "BaseParticleFilterModel"),
        ("yassm.particle_filter", "GenericParticleFilterStateSpaceModelFit", "ParticleFilterFit"),
        (
            "yassm.particle_filter",
            "GeneralLinearGaussianParticleFilterStateSpaceModel",
            "LinearGaussianParticleFilterModel",
        ),
        (
            "yassm.particle_filter",
            "GeneralNonLinearBootstrapParticleFilterStateSpaceModel",
            "NonLinearParticleFilterModel",
        ),
        (
            "yassm.particle_filter",
            "GeneralNonLinearBootstrapParticleFitlerStateSpaceModel",
            "NonLinearParticleFilterModel",
        ),
        ("yassm.lib.qmc.hilbert_curve_full_range", "hilber_sort_inplace_func", "hilbert_sort_inplace_func"),
    ],
)
def test_deprecated_aliases(module: str, old: str, new: str) -> None:
    mod = importlib.import_module(module)
    with pytest.warns(DeprecationWarning, match=old):
        assert getattr(mod, old) is getattr(mod, new)


def test_unknown_attribute_raises() -> None:
    import yassm.particle_filter

    with pytest.raises(AttributeError, match="no attribute"):
        yassm.particle_filter.NoSuchThing  # noqa: B018


def test_fit_accepts_readonly_input() -> None:
    """e.g. DataFrame.values is read-only under pandas' Copy-on-Write."""
    from tests import y
    from yassm.linear_gaussian.model.local_level import local_level_model

    y_ro = y.copy()
    y_ro.flags.writeable = False
    fit = local_level_model.fit(y_ro, [1.0, 1.0], 0.0, 1e8)
    np.testing.assert_allclose(fit.log_likelihood, local_level_model.fit(y.copy(), [1.0, 1.0], 0.0, 1e8).log_likelihood)


def test_as_scipy_objective() -> None:
    """scipy.optimize may pass a non-contiguous x to the objective."""
    from numba import jit

    from yassm.util import as_scipy_objective

    @jit("f8(f8[::1], f8)", nopython=True)
    def f(x, a):
        return x.sum() + a

    x = np.arange(10.0)[::2]
    assert not x.flags.c_contiguous
    assert as_scipy_objective(f)(x, 1.0) == 21.0


def test_fit_plot_renders() -> None:
    """The legend labels must be valid mathtext; this only fails when the figure is drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tests import y
    from yassm.linear_gaussian.model.local_level import local_level_model

    fit = local_level_model.fit(y, [1.0, 1.0], 0.0, 1e8)
    fig, _ = fit.plot(T_forecast=10)
    fig.canvas.draw()
    plt.close(fig)


def test_as_scipy_objective_singular() -> None:
    """A singular matrix at the boundary of the parameter space is +inf, not an error."""
    from numba import jit

    from yassm.util import as_scipy_objective

    @jit("f8(f8[::1])", nopython=True)
    def f(x):
        return np.linalg.inv(np.diag(x)).sum()

    assert as_scipy_objective(f)(np.array([0.0, 1.0])) == np.inf

