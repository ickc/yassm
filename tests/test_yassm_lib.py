"""Compare 1D-grid model to linear Gaussian local ratio model.

So far it is stationary.
"""

from __future__ import annotations

from logging import getLogger

import numpy as np

from yassm.lib import random_choice_weighted

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
