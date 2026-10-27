"""Some utilities, e.g. setup logging."""

from __future__ import annotations

import logging
import os
from logging import Formatter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import numpy.typing as npt

# never emit "No handlers could be found" style warnings from the library
logging.getLogger("yassm").addHandler(logging.NullHandler())


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


def as_scipy_objective(func: Callable[..., float]) -> Callable[..., float]:
    """Wrap a jitted objective function for ``scipy.optimize.minimize``.

    - ``x`` is passed as a C-contiguous float64 array:
      ``scipy.optimize.minimize`` may call the objective with a non-contiguous ``x``,
      which Numba functions with ``f8[::1]`` signatures reject.
    - A singular matrix (e.g. a zero variance on the boundary of the parameter space,
      which the optimizer may probe) gives ``+inf`` instead of raising ``LinAlgError``,
      so that the optimizer backs off.
    """

    def wrapped(x: npt.ArrayLike, *args: Any) -> float:
        try:
            return func(np.ascontiguousarray(x, dtype=np.float64), *args)
        except np.linalg.LinAlgError:
            return np.inf

    return wrapped
