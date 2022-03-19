"""Compare 1D-grid model to linear Gaussian local ratio model.

So far it is stationary.
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from numba import jit

from yassm.grid.one_dimension import General1DStateSpaceModel
from yassm.lib.grid.one_dimension import prob_normal
from yassm.linear_gaussian.model.local_level import local_level_model

from tests import y

if TYPE_CHECKING:
    import numpy.typing as npt

    from yassm.lib import FloatMaybeArray

logger = getLogger(__name__)


@jit("f8(f8, f8, f8[::1], f8)", nopython=True, nogil=True, cache=True)
def transition_func(
    x_t: float,
    x_tMinus1: float,
    params: npt.NDArray[np.float64],
    Δx: float,
) -> float:
    """
    X_t = A_t X_(t-1) + U_t
    U_t ~ N(0, Σ_t)
    """
    σ_ε_sq, σ_Z_sq = params
    return prob_normal(x_t, x_tMinus1, σ_Z_sq, Δx)


@jit("f8(f8, f8, f8[::1], f8)", nopython=True, nogil=True, cache=True)
def design_func(
    y_t: float,
    x_t: float,
    params: npt.NDArray[np.float64],
    Δx: float,
) -> float:
    """
    Y_t = B_t X_t + V_t
    V_t ~ N(0, R_t)
    """
    σ_ε_sq, σ_Z_sq = params
    return prob_normal(y_t, x_t, σ_ε_sq, Δx)


class TestGrid(TestCase):
    x_0 = -3.0
    x_f = 3.0
    G = 100
    params0 = np.array([1.0, 0.01])

    def setUp(self):
        x_0 = self.x_0
        x_f = self.x_f
        G = self.G
        params0 = self.params0

        mod = General1DStateSpaceModel(
            2,
            transition_func,
            design_func,
            x_0,
            x_f,
            G,
            param_names=["σ_ε_sq", "σ_Z_sq"],
        )
        self.res = mod.fit(
            y, params0, np.full(G, 1.0 / G), mle_n_iter=1, mle_kwargs={"bounds": ((0.0, None), (0.0, None))}
        )
        self.ref = local_level_model.fit(y, params0, 0.0, np.infty, mle_n_iter=100)

    def test_mle(self):
        np.testing.assert_allclose(self.res.params, self.ref.params)
