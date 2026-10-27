from __future__ import annotations

import numpy as np

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
