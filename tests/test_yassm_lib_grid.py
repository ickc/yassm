"""Compare 1D-grid model to linear Gaussian local ratio model.

So far it is stationary.
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from numba import jit

from yassm.lib import normal_1d
from yassm.lib.grid.one_dimension import (
    filter_update_inplace,
    get_grid_to_position,
    one_step_ahead_inplace,
    smoothing_backward_recursion_inplace,
)
from yassm.lib.linear_gaussian.model.local_ratio import filter_update as local_ratio_filter_update
from yassm.lib.linear_gaussian.model.local_ratio import one_step_ahead as local_ratio_one_step_ahead
from yassm.lib.linear_gaussian.model.local_ratio import (
    smoothing_backward_recursion as local_ratio_smoothing_backward_recursion,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from yassm.lib import FloatMaybeArray

logger = getLogger(__name__)


@pytest.mark.parametrize(
    "x_0, x_f, G",
    (
        (0.0, 1.0, 100),
        (-19.0, 72.0, 1000),
    ),
)
def test_get_grid_to_position(x_0, x_f, G):
    grid_to_position = get_grid_to_position(x_0, x_f, G)
    np.testing.assert_array_equal(grid_to_position(np.arange(G)), np.linspace(x_0, x_f, G))


@jit("f8(f8, f8, f8[::1])", nopython=True, nogil=True, cache=True)
def transition_func(
    x_t: float,
    x_tMinus1: float,
    params: npt.NDArray[np.float64],
) -> float:
    """
    X_t = A_t X_(t-1) + U_t
    U_t ~ N(0, Σ_t)
    """
    σ_ε_sq, σ_Z_sq, φ = params
    return normal_1d(x_t, φ * x_tMinus1, σ_Z_sq)


@jit("f8(f8, f8, f8[::1])", nopython=True, nogil=True, cache=True)
def design_func(
    y_t: float,
    x_t: float,
    params: npt.NDArray[np.float64],
) -> float:
    """
    Y_t = B_t X_t + V_t
    V_t ~ N(0, R_t)
    """
    σ_ε_sq, σ_Z_sq, φ = params
    return normal_1d(y_t, x_t, σ_ε_sq)


@pytest.mark.parametrize(
    "σ_ε_sq,σ_Z_sq,φ,G,n_σ,m_sMinus1_given_t,Q_sMinus1_given_t",
    (
        (0.1, 4.0, 1.0, 1000, 6.0, 2.0, 1.5),
        (0.2, 1.7, 0.9, 1500, 6.0, 1.3, 1.9),
    ),
)
def test_one_step_ahead(
    σ_ε_sq: float,
    σ_Z_sq: float,
    φ: float,
    G: int,
    n_σ: float,
    m_sMinus1_given_t: float,
    Q_sMinus1_given_t: float,
    raise_assertion_error: bool = True,
) -> tuple[npt.NDArray[np.float64], float, float, npt.NDArray[np.float64], float, float, npt.NDArray[np.float64],]:
    params = np.array([σ_ε_sq, σ_Z_sq, φ])
    σ_Z = np.sqrt(σ_Z_sq)
    x_f = n_σ * σ_Z
    x_0 = -x_f

    x: npt.NDArray[np.float64] = np.linspace(x_0, x_f, G)

    p_sMinus1_given_t = normal_1d(
        x,
        m_sMinus1_given_t,
        Q_sMinus1_given_t,
    )
    p_s_given_t = np.zeros(G)
    one_step_ahead_inplace(
        p_s_given_t,
        params,
        transition_func,
        get_grid_to_position(x_0, x_f, G),
        p_sMinus1_given_t,
    )
    m_s_given_t_hat = (p_s_given_t * x).sum()
    Q_s_given_t_hat = (p_s_given_t * np.square(x - m_s_given_t_hat)).sum()

    m_s_given_t, Q_s_given_t = local_ratio_one_step_ahead(
        φ,
        σ_Z_sq,
        m_sMinus1_given_t,
        Q_sMinus1_given_t,
    )

    p_s_given_t_ref = normal_1d(
        x,
        m_s_given_t,
        Q_s_given_t,
    )
    p_s_given_t_ref /= p_s_given_t_ref.sum()

    try:
        np.testing.assert_allclose(m_s_given_t_hat, m_s_given_t, rtol=1e-2)
        np.testing.assert_allclose(Q_s_given_t_hat, Q_s_given_t, rtol=1e-2)
        np.testing.assert_allclose(
            p_s_given_t,
            p_s_given_t_ref,
            rtol=0.01,
        )
    except AssertionError as e:
        if raise_assertion_error:
            raise e
        else:
            logger.warning(e)

    return (
        x,
        m_s_given_t_hat,
        Q_s_given_t_hat,
        p_s_given_t,
        m_s_given_t,
        Q_s_given_t,
        p_s_given_t_ref,
    )  # type: ignore[return-value]


@pytest.mark.parametrize(
    "σ_ε_sq,σ_Z_sq,φ,G,n_σ,m_t_given_tMinus1,Q_t_given_tMinus1,y_t",
    (
        (0.1, 4.0, 1.0, 1000, 3.0, 2.0, 1.5, 1.3),
        (0.2, 1.7, 0.9, 1500, 5.0, 1.3, 1.9, 2.1),
    ),
)
def test_filter_update(
    σ_ε_sq: float,
    σ_Z_sq: float,
    φ: float,
    G: int,
    n_σ: float,
    m_t_given_tMinus1: float,
    Q_t_given_tMinus1: float,
    y_t: float,
    raise_assertion_error: bool = True,
) -> tuple[npt.NDArray[np.float64], float, float, npt.NDArray[np.float64], float, float, npt.NDArray[np.float64],]:
    params = np.array([σ_ε_sq, σ_Z_sq, φ])
    σ_Z = np.sqrt(σ_Z_sq)
    σ_ε = np.sqrt(σ_ε_sq)
    x_f = n_σ * σ_Z
    x_0 = -x_f

    x = np.linspace(x_0, x_f, G)

    p_t_given_tMinus1 = normal_1d(
        x,
        m_t_given_tMinus1,
        Q_t_given_tMinus1,
    )
    p_t_given_t = np.zeros(G)
    filter_update_inplace(
        p_t_given_t,
        y_t,
        params,
        design_func,
        get_grid_to_position(x_0, x_f, G),
        p_t_given_tMinus1,
    )
    m_t_given_t_hat = (p_t_given_t * x).sum()
    Q_t_given_t_hat = (p_t_given_t * np.square(x - m_t_given_t_hat)).sum()

    m_t_given_t, Q_t_given_t = local_ratio_filter_update(
        y_t,
        σ_ε_sq,
        m_t_given_tMinus1,
        Q_t_given_tMinus1,
    )

    p_t_given_t_ref = normal_1d(
        x,
        m_t_given_t,
        Q_t_given_t,
    )
    p_t_given_t_ref /= p_t_given_t_ref.sum()

    try:
        np.testing.assert_allclose(m_t_given_t_hat, m_t_given_t)
        np.testing.assert_allclose(Q_t_given_t_hat, Q_t_given_t)

        np.testing.assert_allclose(
            p_t_given_t,
            p_t_given_t_ref,
            rtol=0.01,
        )
    except AssertionError as e:
        if raise_assertion_error:
            raise e
        else:
            logger.warning(e)

    return (
        x,
        m_t_given_t_hat,
        Q_t_given_t_hat,
        p_t_given_t,
        m_t_given_t,
        Q_t_given_t,
        p_t_given_t_ref,
    )  # type: ignore[return-value]


@pytest.mark.parametrize(
    "σ_ε_sq,σ_Z_sq,φ,G,n_σ,m_s_given_s,Q_s_given_s,m_sPlus1_given_t,Q_sPlus1_given_t,m_sPlus1_given_s,Q_sPlus1_given_s",
    (
        (0.1, 4.0, 1.0, 1000, 3.0, 2.0, 1.5, 1.3, 2.1, 4.2, 1.1),
        (0.2, 1.7, 0.9, 1500, 5.0, 1.3, 1.9, 2.1, 0.7, 1.34, 3.33),
    ),
)
def test_smoothing_backward_recursion_inplace(
    σ_ε_sq: float,
    σ_Z_sq: float,
    φ: float,
    G: int,
    n_σ: float,
    m_s_given_s: float,
    Q_s_given_s: float,
    m_sPlus1_given_t: float,
    Q_sPlus1_given_t: float,
    m_sPlus1_given_s: float,
    Q_sPlus1_given_s: float,
    raise_assertion_error: bool = True,
) -> tuple[npt.NDArray[np.float64], float, float, npt.NDArray[np.float64], float, float, npt.NDArray[np.float64],]:
    params = np.array([σ_ε_sq, σ_Z_sq, φ])
    σ_Z = np.sqrt(σ_Z_sq)
    σ_ε = np.sqrt(σ_ε_sq)
    x_f = n_σ * σ_Z
    x_0 = -x_f
    R = σ_ε_sq
    Σ = σ_Z_sq

    x = np.linspace(x_0, x_f, G)

    p_s_given_s = normal_1d(
        x,
        m_s_given_s,
        Q_s_given_s,
    )
    p_s_given_s /= p_s_given_s.sum()
    p_sPlus1_given_t = normal_1d(
        x,
        m_sPlus1_given_t,
        Q_sPlus1_given_t,
    )
    p_sPlus1_given_t /= p_sPlus1_given_t.sum()
    p_sPlus1_given_s = normal_1d(
        x,
        m_sPlus1_given_s,
        Q_sPlus1_given_s,
    )
    p_sPlus1_given_s /= p_sPlus1_given_s.sum()

    grid_to_position = get_grid_to_position(x_0, x_f, G)
    p_s_given_t = np.zeros(G)
    smoothing_backward_recursion_inplace(
        p_s_given_t,
        params,
        transition_func,
        grid_to_position,
        p_s_given_s,
        p_sPlus1_given_t,
        p_sPlus1_given_s,
    )
    m_s_given_t_hat = (p_s_given_t * x).sum()
    Q_s_given_t_hat = (p_s_given_t * np.square(x - m_s_given_t_hat)).sum()

    m_s_given_t, Q_s_given_t = local_ratio_smoothing_backward_recursion(
        φ,
        Σ,
        m_s_given_s,
        Q_s_given_s,
        m_sPlus1_given_t,
        Q_sPlus1_given_t,
    )

    p_s_given_t_ref = normal_1d(
        x,
        m_s_given_t,
        Q_s_given_t,
    )
    p_s_given_t_ref /= p_s_given_t_ref.sum()

    try:
        np.testing.assert_allclose(m_s_given_t_hat, m_s_given_t)
        np.testing.assert_allclose(Q_s_given_t_hat, Q_s_given_t)
        np.testing.assert_allclose(
            p_s_given_t,
            p_s_given_t_ref,
            rtol=0.01,
        )
    except AssertionError as e:
        if raise_assertion_error:
            raise e
        else:
            logger.warning(e)

    return x, m_s_given_t_hat, Q_s_given_t_hat, p_s_given_t, m_s_given_t, Q_s_given_t, p_s_given_t_ref  # type: ignore[return-value]


def res_to_df(
    x: npt.NDArray[np.float64],
    m_hat: float,
    Q_hat: float,
    res: npt.NDArray[np.float64],
    m: float,
    Q: float,
    ref: npt.NDArray[np.float64],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the above return values to a DataFrame."""
    df = pd.DataFrame(
        {
            "res": res,
            "ref": ref,
        },
        index=x,
    )
    df_summary = pd.DataFrame(
        [
            [m_hat, Q_hat],
            [m, Q],
        ],
        index=["estimated", "exact"],
        columns=["m", "Q"],
    )
    return df, df_summary
