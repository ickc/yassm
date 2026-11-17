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

    from jaxtyping import Float64 as Float64
else:
    # jaxtyping is not needed to run yassm, whose annotations are not evaluated at import:
    # only to check them, here or statically, and to render them in the documentation.
    try:
        from jaxtyping import Float64
    except ImportError:

        class Float64:
            """Stand-in for `jaxtyping.Float64` without jaxtyping: ``Float64[np.ndarray, "T d"]`` is ``np.ndarray``."""

            def __class_getitem__(cls, item: tuple[Any, str]) -> Any:
                return item[0]


FuncT = TypeVar("FuncT", bound="Callable[..., Any]")

#: Whether `typechecked` checks, set by the ``YASSM_TYPECHECK`` environment variable at import.
TYPECHECK: bool = os.environ.get("YASSM_TYPECHECK", "").strip().lower() not in ("", "0", "false", "no", "off")


def typechecked(func: FuncT) -> FuncT:
    """Check the dtypes and shapes of the arguments and return value of `func` if `TYPECHECK`, else return `func`.

    The dimensions bind within each call, e.g. a tuple of ``"T n"`` arrays must all have the same shape.
    Checking requires `jaxtyping <https://docs.kidger.site/jaxtyping/>`_ and
    `beartype <https://beartype.readthedocs.io>`_, which yassm does not depend on otherwise.
    It cannot apply to Numba-jitted functions, which ignore annotations.
    """
    if not TYPECHECK:
        return func
    try:
        from beartype import beartype
        from jaxtyping import jaxtyped
    except ImportError as error:
        raise ImportError(
            "YASSM_TYPECHECK is set: checking the array annotations needs jaxtyping and beartype."
        ) from error

    return jaxtyped(typechecker=beartype)(func)
