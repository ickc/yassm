"""yassm: Yet Another State Space Model package.

The main classes and ready-made models are available from the top level, e.g.
``yassm.LinearGaussianModel`` or ``yassm.local_level_model``.
They are imported lazily, as importing them compiles their Numba kernels (or loads them from the cache).
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any

from .util import setup_logging

__version__ = "0.1.0"

# A library should not configure logging on import unless asked to:
# set YASSMLOGLEVEL (e.g. INFO) or call yassm.util.setup_logging() to see yassm's logs.
if os.environ.get("YASSMLOGLEVEL"):
    setup_logging()

_EXPORTS: dict[str, str] = {
    "LinearGaussianModel": "yassm.linear_gaussian",
    "LinearGaussianFit": "yassm.linear_gaussian",
    "local_level_model": "yassm.linear_gaussian.model.local_level",
    "local_linear_model": "yassm.linear_gaussian.model.local_linear",
    "local_ratio_model": "yassm.linear_gaussian.model.local_ratio",
    "moving_average_1_model": "yassm.linear_gaussian.model.moving_average_1",
    "GridFilterModel": "yassm.grid.one_dimension",
    "GridFilterFit": "yassm.grid.one_dimension",
    "LinearGaussianParticleFilterModel": "yassm.particle_filter",
    "NonLinearParticleFilterModel": "yassm.particle_filter",
    "ParticleFilterFit": "yassm.particle_filter",
    "HaltonSequence": "yassm.qmc.random",
    "HilbertCurveInt": "yassm.qmc.hilbert_curve",
    "HilbertCurveFloat01": "yassm.qmc.hilbert_curve",
    "HilbertCurveFloat": "yassm.qmc.hilbert_curve",
}

__all__ = ["__version__", "setup_logging", *_EXPORTS]

if TYPE_CHECKING:
    from .grid.one_dimension import GridFilterFit, GridFilterModel
    from .linear_gaussian import LinearGaussianFit, LinearGaussianModel
    from .linear_gaussian.model.local_level import local_level_model
    from .linear_gaussian.model.local_linear import local_linear_model
    from .linear_gaussian.model.local_ratio import local_ratio_model
    from .linear_gaussian.model.moving_average_1 import moving_average_1_model
    from .particle_filter import LinearGaussianParticleFilterModel, NonLinearParticleFilterModel, ParticleFilterFit
    from .qmc.hilbert_curve import HilbertCurveFloat, HilbertCurveFloat01, HilbertCurveInt
    from .qmc.random import HaltonSequence


def __getattr__(name: str) -> Any:
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
