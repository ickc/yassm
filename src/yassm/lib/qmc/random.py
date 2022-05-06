"""QMC point-sets or sequences as pseudo-random number generators.

All point-sets or sequences are Pseudo Unif(0, 1) exclusive.

We do not include 0 to avoid singularity,
where the point-set/sequence would be more symmetric too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt


@jit("f8[::1](i8)", nopython=True, nogil=True, cache=True)
def midpoint_pointset(N: int) -> npt.NDArray[np.float64]:
    """Returns the midpoint point-set which minimize star-discrepancy in 1D."""
    res = np.empty(N)
    inv_2N = 1.0 / (2 * N)
    return np.linspace(inv_2N, 1.0 - inv_2N, N)
