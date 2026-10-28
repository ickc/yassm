"""Shape-annotated arrays, checked at runtime in the test suite.

Arrays are annotated with `jaxtyping <https://docs.kidger.site/jaxtyping/>`_, e.g. ``Float64[np.ndarray, "T d"]``
is a float64 array of shape ``(T, d)``; a single dimension is written with a leading space, e.g. ``" T"``,
so that linters do not take it for a name. The dimensions are named:

``T``
    time steps
``n``
    dimension of the observations
``d``
    dimension of the states
``p``
    number of parameters
``N``
    number of draws: samples, particles, or iterations of a chain
``K``
    number of parameter sets, e.g. a grid of parameters
``G``
    number of grid points of a grid filter
``T_forecast``
    forecast time steps
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

FuncT = TypeVar("FuncT", bound="Callable[..., Any]")

#: Whether `typechecked` checks, set by the ``YASSM_TYPECHECK`` environment variable at import.
TYPECHECK: bool = os.environ.get("YASSM_TYPECHECK", "") not in ("", "0")


def typechecked(func: FuncT) -> FuncT:
    """Check the dtypes and shapes of the arguments and return value of `func` if `TYPECHECK`, else return `func`.

    The dimensions bind within each call, e.g. a tuple of ``"T n"`` arrays must all have the same shape.
    Checking requires `beartype <https://beartype.readthedocs.io>`_.
    It cannot apply to Numba-jitted functions, which ignore annotations.
    """
    if not TYPECHECK:
        return func
    from beartype import beartype
    from jaxtyping import jaxtyped

    return jaxtyped(typechecker=beartype)(func)
