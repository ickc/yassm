from __future__ import annotations

import numpy as np

from yassm.lib import sqrt_diag


def test_sqrt_diag() -> None:
    array = np.arange(12, dtype=np.float64).reshape((3, 2, 2))
    res = sqrt_diag(array)
    ref = np.sqrt(
        np.array(
            [
                [0, 3],
                [4, 7],
                [8, 11],
            ]
        )
    )
    np.testing.assert_array_equal(res, ref)
