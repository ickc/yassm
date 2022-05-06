from __future__ import annotations

import numpy as np
import pytest

from yassm.lib.qmc.random import midpoint_pointset, van_der_corput_sequence, van_der_corput_sequence_slow


@pytest.mark.parametrize(
    "N",
    (3, 5, 11, 31),
)
def test_midpoint_pointset(N) -> None:
    np.testing.assert_allclose(midpoint_pointset(N) * (2 * N), np.arange(1, 2 * N + 1, 2))


@pytest.mark.parametrize(
    "N, base",
    (
        (1000, 2),
        (1000, 3),
        (1000, 11),
        (1000, 31),
    ),
)
def test_van_der_corput_sequence(N, base) -> None:
    np.testing.assert_allclose(
        van_der_corput_sequence(N, base),
        van_der_corput_sequence_slow(N, base),
    )
