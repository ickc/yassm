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


def test_random_choice_weighted_fewer_samples_than_bins():
    """The out-of-range index must be clamped to the number of bins, not the number of samples."""
    x = np.arange(3.0).reshape(3, 1)
    # all the mass is on x[1], and searchsorted returns 1 == size for every draw
    res = random_choice_weighted(x, 1, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(res, [[1.0]])


def test_random_choice_weighted_more_samples_than_bins():
    """u >= b[-1] (here via an infinite weight) must not read past the end of x."""
    x = np.arange(2.0).reshape(2, 1)
    res = random_choice_weighted(x, 5, np.array([1.0, np.inf]))
    np.testing.assert_array_equal(res, np.ones((5, 1)))
