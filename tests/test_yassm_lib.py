"""Compare 1D-grid model to linear Gaussian local ratio model.

So far it is stationary.
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from numba import jit

from yassm.lib import random_choice_weighted
from yassm.lib.linear_gaussian.model.local_ratio import filter_update as local_ratio_filter_update
from yassm.lib.linear_gaussian.model.local_ratio import one_step_ahead as local_ratio_one_step_ahead
from yassm.lib.linear_gaussian.model.local_ratio import (
    smoothing_backward_recursion as local_ratio_smoothing_backward_recursion,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from yassm.lib import FloatMaybeArray

logger = getLogger(__name__)


def test_random_choice_weighted():
    x = np.arange(8, dtype=np.float64).reshape(4, 2)
    p = np.linspace(0.1, 0.4, 4)
    N = 10000

    # 10σ
    rtol = 10.0 / np.sqrt(N)

    # test un-normalized
    res = random_choice_weighted(x, N, p * 10.0)
    temp = np.unique(res, axis=0, return_counts=True)
    np.testing.assert_array_equal(temp[0], x)
    np.testing.assert_allclose(temp[1] / N, p, rtol=rtol)
