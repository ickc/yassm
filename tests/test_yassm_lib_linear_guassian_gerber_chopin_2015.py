from __future__ import annotations

import numpy as np

from yassm.lib.linear_gaussian.model.gerber_chopin_2015 import get_F


def test_F():
    d = 10
    α = 0.1

    i = np.arange(d)
    i, j = np.meshgrid(i, i)
    exp = np.abs(i - j)
    ref = np.power(α, exp)
    np.testing.assert_array_equal(get_F(α, d), ref)
