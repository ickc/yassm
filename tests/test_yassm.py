from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import TestCase

import numpy as np
import statsmodels.tsa.api as tsa

from yassm.lib import sqrt_diag
from yassm.linear_gaussian.model.local_level import local_level_model

from tests import y

if TYPE_CHECKING:
    from yassm.linear_gaussian import GeneralLinearGaussianStateSpaceModelFit


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
