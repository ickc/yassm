from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    import numpy.typing as npt


@jit("float64[::1](float64[::1])", nogil=True, nopython=True)
def fn(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.sin(15.0 * x) + 3.0 * np.exp(-0.5 * np.square(x)) + 0.5 * np.square(x - 0.5)


@jit(nogil=True, nopython=True)
def set_seed(value: int) -> None:
    """
    See https://github.com/numba/numba/issues/6002#issuecomment-660152030
    """
    np.random.seed(value)


rng = np.random.default_rng(314)
y = fn(np.linspace(0.0, 1.0, 500)) + rng.normal(scale=1.0, size=500)
y_nan = y.copy()
y_nan[::10] = np.nan
