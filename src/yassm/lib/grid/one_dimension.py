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

    from .. import FloatMaybeArray


def get_grid_to_position(
    x_0: float,
    x_f: float,
    G: int,
) -> Callable[[int], float]:
    """Compile grid_to_position given args.

    Signiture is similar to `numpy.linspace(x_0, x_f, G)`.
    """

    @jit(["f8(i8)", "f8[::1](i8[::1])"], nopython=True, nogil=True, cache=False)
    def grid_to_position(g: int) -> float:

        # g is from 0 to G - 1
        return g * ((x_f - x_0) / (G - 1)) + x_0

    return grid_to_position


@jit("(f8[::1], f8[::1], f8[::1])", nopython=True, nogil=True, cache=True)
def p_to_mQ_inplace(
    mQ: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
):
    m = (p * x).sum()
    mQ[0] = m
    mQ[1] = (p * np.square(x - m)).sum()


@jit("f8[:, ::1](f8[:, ::1], f8[::1])", nopython=True, nogil=True, cache=True, parallel=True)
def ps_to_mQ(
    ps: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    size, Q = ps.shape
    mQs = np.empty((size, 2))
    for t in prange(size):
        p_to_mQ_inplace(
            mQs[t],
            ps[t],
            x,
        )
    return mQs


@jit(nopython=True, nogil=True, cache=False, parallel=True)
def one_step_ahead_inplace(
    p_s_given_t: npt.NDArray[np.float64],
    params: npt.NDArray[np.float64],
    transition_func: Callable[[float, float, npt.NDArray[np.float64]], float],
    grid_to_position: Callable[[int], float],
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
    grid_to_position: Callable[[int], float],
    p_t_given_tMinus1: npt.NDArray[np.float64],
) -> None:
    """Filter update inplace.

    :param p_t_given_t: output array, must be intialized with zeros.
    :param design_func: a function of (y_t, x_t, params) returning the
        density at y_t given X_t = x_t.
    :param grid_to_position: a function from grid index to grid position.
    """
    G = p_t_given_tMinus1.size

    for g in prange(G):
        x = grid_to_position(g)
        p_t_given_t[g] = design_func(y_t, x, params) * p_t_given_tMinus1[g]
    # normalize
    p_t_given_t /= p_t_given_t.sum()


@jit(nopython=True, nogil=True, cache=False, parallel=True)
def smoothing_backward_recursion_inplace(
    p_s_given_t: npt.NDArray[np.float64],
    params: npt.NDArray[np.float64],
    transition_func: Callable[[float, float, npt.NDArray[np.float64]], float],
    grid_to_position: Callable[[int], float],
    p_s_given_s: npt.NDArray[np.float64],
    p_sPlus1_given_t: npt.NDArray[np.float64],
    p_sPlus1_given_s: npt.NDArray[np.float64],
) -> None:
    """Smoothing backward recursion inplace.

    :param p_s_given_t: output array, must be intialized with zeros.
    :param transition_func: a function of (x_t, x_tMinus1, params) returning the
        density at x_t given previous step at x_tMinus1.
    :param grid_to_position: a function from grid index to grid position.
    """
    G = p_s_given_t.size

    # we parallelize on g
    # as it is numerically unstable to shuffle the order of summing over g_p
    for g in prange(G):
        x = grid_to_position(g)
        for g_p in range(G):
            x_p = grid_to_position(g_p)
            # avoid underflow problem
            if p_sPlus1_given_s[g_p] != 0.0:
                p_s_given_t[g] += transition_func(x_p, x, params) * p_sPlus1_given_t[g_p] / p_sPlus1_given_s[g_p]
        p_s_given_t[g] *= p_s_given_s[g]
    # normalize
    p_s_given_t /= p_s_given_t.sum()
