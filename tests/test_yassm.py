from __future__ import annotations

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

    assert logging.getLogger("yassm").handlers == []


def test_warnings_are_printed_by_default() -> None:
    """Without any logging configuration, Python prints warnings (not info) to stderr, unless a handler is attached.

    yassm warns e.g. when it ignores an argument or falls back to another MLE method.
    """
    import os
    import subprocess
    import sys

    code = """
import logging
import yassm
logger = logging.getLogger("yassm.linear_gaussian")
logger.info("progress")
logger.warning("falling back")
"""
    env = {key: value for key, value in os.environ.items() if key != "YASSMLOGLEVEL"}
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert res.stderr.strip() == "falling back"


def test_default_mle_method_does_not_warn(caplog) -> None:
    """`mle_method` defaults to "", the default method of the model: only an unknown method is replaced, with a warning."""
    from tests import y
    from yassm.linear_gaussian.model.local_level import local_level_model
    from yassm.linear_gaussian.model.local_linear import local_linear_model
    from yassm.linear_gaussian.model.local_ratio import local_ratio_model

    for model, μ_0, Γ_0 in (
        (local_level_model, 0.0, 1e8),
        (local_ratio_model, 0.0, 1e8),
        (local_linear_model, np.zeros(2), 1e8 * np.eye(2)),
    ):
        for mle_method, replaced in (("", False), ("newton", True)):
            caplog.clear()
            with caplog.at_level("WARNING", logger="yassm"):
                model.fit(y.copy(), None, μ_0, Γ_0, mle_method=mle_method, mle_n_iter=1)
            assert ("not supported" in caplog.text or "only implemented" in caplog.text) is replaced


def test_deprecated_aliases() -> None:
    from yassm.lib.qmc import hilbert_curve_full_range
    from yassm.particle_filter import (
        GeneralNonLinearBootstrapParticleFilterStateSpaceModel,
        GeneralNonLinearBootstrapParticleFitlerStateSpaceModel,
    )

    assert (
        GeneralNonLinearBootstrapParticleFitlerStateSpaceModel is GeneralNonLinearBootstrapParticleFilterStateSpaceModel
    )
    assert hilbert_curve_full_range.hilber_sort_inplace_func is hilbert_curve_full_range.hilbert_sort_inplace_func


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

    @jit("f8(f8[::1])", nopython=True)
    def g(x):
        return np.linalg.cholesky(np.diag(x)).sum()

    assert as_scipy_objective(g)(np.array([-1.0, 1.0])) == np.inf

    # NumPy words it differently from Numba
    assert as_scipy_objective(lambda x: np.linalg.inv(np.diag(x)).sum())(np.array([0.0, 1.0])) == np.inf


def test_as_scipy_objective_zero_variance() -> None:
    """The scalar kernels, e.g. of the local level model, divide by the variances: a zero one is +inf too.

    It raised ZeroDivisionError, e.g. when L-BFGS-B probed a bound of exactly 0.
    """
    from tests import y
    from yassm.linear_gaussian.model.local_level import local_level_model
    from yassm.util import as_scipy_objective

    objective = as_scipy_objective(local_level_model.neg2_log_likelihood)
    args = (y.reshape(-1, 1), np.zeros(1), np.eye(1))
    assert np.isfinite(objective(np.array([1.0, 1.0]), *args))
    assert objective(np.array([0.0, 1.0]), *args) == np.inf


def test_as_scipy_objective_raises_other_errors() -> None:
    """Only a matrix that cannot be inverted at these parameters is +inf: other errors hold for any parameters."""
    from numba import jit

    from yassm.util import as_scipy_objective

    @jit("f8(f8[::1])", nopython=True)
    def wrong_shape(x):
        return np.linalg.inv(np.ones((2, 3)) * x[0]).sum()

    @jit("f8(f8[::1], f8)", nopython=True)
    def with_data(x, y):
        return np.linalg.solve(np.diag(x), np.full(x.size, y)).sum()

    x = np.array([1.0, 2.0])
    with pytest.raises(np.linalg.LinAlgError, match="square"):
        as_scipy_objective(wrong_shape)(x)
    with pytest.raises(np.linalg.LinAlgError, match="NaN"):
        as_scipy_objective(with_data)(x, np.nan)
    # but the optimizer may try non-finite parameters, e.g. TNC after an infinite objective
    assert as_scipy_objective(with_data)(np.array([1.0, np.nan]), 0.0) == np.inf
