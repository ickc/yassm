"""Some utilities, e.g. setup logging."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from logging import Formatter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import numpy.typing as npt

#: ``scipy.optimize`` bounds for a variance. Strictly positive: at exactly zero, covariance matrices can be
#: singular, making the objective infinite, and L-BFGS-B then stops early while reporting success.
VARIANCE_BOUNDS: tuple[float, None] = (float(np.finfo(np.float64).eps), None)


def setup_logging(
    name: str = "yassm",
    level: str = "INFO",
    env_var: str = "YASSMLOGLEVEL",
    fmt: str = "%(name)s %(levelname)s %(message)s",
    Handler: type[logging.Handler] = logging.StreamHandler,
    Formatter: type[logging.Formatter] = Formatter,
) -> logging.Logger:
    """Setup logging.

    Parameters
    ----------
    env_var
        if not empty, set log level from this environment variable.
    """
    logger = logging.getLogger(name)
    handler = Handler()
    handler.setFormatter(Formatter(fmt))
    logger.addHandler(handler)
    if env_var:
        try:
            logger.setLevel(level=(_level := os.environ.get(env_var, level)))
        except ValueError:
            logger.setLevel(level=level)
            logger.error(
                "%s=%s is not a valid logging level, set to default %s.",
                env_var,
                _level,
                level,
            )
    else:
        logger.setLevel(level=level)
    return logger


def as_c_array(
    array: npt.ArrayLike,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.generic]:
    """Return `array` as a C-contiguous (ndim >= 1), writeable array of `dtype`, copying only if needed.

    Numba functions with explicit signatures reject read-only arrays, which are
    e.g. returned by ``DataFrame.values`` under pandas' Copy-on-Write.
    """
    return np.require(np.ascontiguousarray(array, dtype=dtype), requirements="W")


#: How NumPy and Numba word the `LinAlgError` of a matrix that is singular or not positive definite.
_NOT_INVERTIBLE = ("singular", "positive definite")


def as_scipy_objective(func: Callable[..., float]) -> Callable[..., float]:
    """Wrap a jitted objective function for ``scipy.optimize.minimize``.

    - ``x`` is passed as a C-contiguous float64 array:
      ``scipy.optimize.minimize`` may call the objective with a non-contiguous ``x``,
      which Numba functions with ``f8[::1]`` signatures reject.
    - A singular or not positive definite matrix (e.g. a zero variance on the boundary of the parameter space,
      which the optimizer may probe) gives ``+inf`` instead of raising ``LinAlgError``,
      so that the optimizer backs off. So does a zero variance in the scalar kernels, e.g. of the local level model,
      which divide by it (``ZeroDivisionError``). So does any ``LinAlgError`` at a non-finite ``x``,
      which some optimizers try after an infinite objective.

    Any other ``LinAlgError`` is raised: matrices of the wrong shape, or NaN in the data,
    are errors at every ``x``, which would otherwise only show as an optimization that fails.
    """

    def wrapped(x: npt.ArrayLike, *args: Any) -> float:
        x = np.ascontiguousarray(x, dtype=np.float64)
        try:
            return func(x, *args)
        except np.linalg.LinAlgError as error:
            message = str(error).lower()
            if any(text in message for text in _NOT_INVERTIBLE) or not np.isfinite(x).all():
                return np.inf
            raise
        except ZeroDivisionError:
            return np.inf

    return wrapped


def deprecated_aliases(module: str, aliases: dict[str, str]) -> Callable[[str], Any]:
    """Return a module ``__getattr__`` resolving deprecated names with a `DeprecationWarning`.

    Parameters
    ----------
    module
        ``__name__`` of the module defining the new names.
    aliases
        Mapping from each deprecated name to its replacement.

    Examples
    --------
    >>> __getattr__ = deprecated_aliases(__name__, {"OldName": "NewName"})  # doctest: +SKIP
    """

    def __getattr__(name: str) -> Any:
        try:
            new = aliases[name]
        except KeyError:
            raise AttributeError(f"module {module!r} has no attribute {name!r}") from None
        warnings.warn(
            f"{module}.{name} is deprecated, use {module}.{new} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(sys.modules[module], new)

    return __getattr__
