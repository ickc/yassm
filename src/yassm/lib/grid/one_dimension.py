"""Naïve grid method.

The quantity to pass around is a 1d-array
representing the density over a fixed interval throughout.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit, prange

if TYPE_CHECKING:
    from typing import Callable

    import numpy.typing as npt


def get_grid_to_position(x_0, x_f, G):
    """Compile grid_to_position given args.

    Signiture is similar to `numpy.linspace(x_0, x_f, G)`.
    """

    @jit
    def grid_to_position(g):

        # g is from 0 to G - 1
        return g * ((x_f - x_0) / (G - 1)) + x_0

    return grid_to_position


@jit(nopython=True, nogil=True, cache=False, parallel=True)
def one_step_ahead_inplace(
    p_s_given_t: npt.NDArray[np.float64],
    params: npt.NDArray[np.float64],
    transition_func: Callable[[float, float, npt.NDArray[np.float64]], float],
    grid_to_position: Callable[[float], float],
    p_sMinus1_given_t: npt.NDArray[np.float64],
) -> None:
    """One-step ahead prediction inplace.

    :param p_s_given_t: output array, must be intialized with zeros.
    :param transition_func: a function of (x_t, x_tMinus1, params) returning the
        density at x_t given previous step at x_tMinus1.
    :param grid_to_position: a function from grid index to grid position.
    """
    G = p_sMinus1_given_t.size

    # we parallelize on g
    # as it is numerically unstable to shuffle the order of summing over g_p
    for g in prange(G):
        x = grid_to_position(g)
        for g_p in range(G):
            x_p = grid_to_position(g_p)
            p_s_given_t[g] += transition_func(x, x_p, params) * p_sMinus1_given_t[g_p]
    # normalize
    p_s_given_t /= p_s_given_t.sum()


@jit(nopython=True, nogil=True, cache=False, parallel=True)
def filter_update_inplace(
    p_t_given_t: npt.NDArray[np.float64],
    y_t: float,
    params: npt.NDArray[np.float64],
    design_func: Callable[[float, float, npt.NDArray[np.float64]], float],
    grid_to_position: Callable[[float], float],
    p_t_given_tMinus1: npt.NDArray[np.float64],
) -> None:
    """Filter update inplace.

    :param p_t_given_t: output array, must be intialized with zeros.
    :param design_func: a function of (y_t, x_t, params) returning the
        density at y_t given X_t = x_t.
    :param grid_to_position: a function from grid index to grid position.
    """
    G = p_t_given_tMinus1.size

    for g in range(G):
        x = grid_to_position(g)
        p_t_given_t[g] = design_func(y_t, x, params) * p_t_given_tMinus1[g]
    # normalize
    p_t_given_t /= p_t_given_t.sum()
