from __future__ import annotations

import numpy as np
import pytest

from yassm.lib.qmc.random import midpoint_pointset, van_der_corput_sequence


def test_midpoint_pointset(N=10) -> None:
    np.testing.assert_allclose(midpoint_pointset(N) * (2 * N), np.arange(1, 2 * N + 1, 2))


def test_van_der_corput_sequence(N=9, base=2) -> None:
    np.testing.assert_array_equal(
        van_der_corput_sequence(N, base),
        np.array([0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625, 0.5625]),
    )
