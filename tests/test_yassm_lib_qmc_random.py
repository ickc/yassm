from __future__ import annotations

import numpy as np
import pytest

from yassm.lib.qmc.random import midpoint_pointset


def test_midpoint_pointset(N=10) -> None:
    np.testing.assert_allclose(midpoint_pointset(N) * (2 * N), np.arange(1, 2 * N + 1, 2))
