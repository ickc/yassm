"""Compare 1D-grid model to linear Gaussian local ratio model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numba import jit, vectorize

from yassm.lib import normal_1d
from yassm.lib.grid.one_dimension import filter_update_inplace, get_grid_to_position, one_step_ahead_inplace
from yassm.lib.linear_gaussian.model.local_ratio import filter_update as local_ratio_filter_update
from yassm.lib.linear_gaussian.model.local_ratio import one_step_ahead as local_ratio_one_step_ahead


@jit
def transition_func(
    x_t,
    x_tMinus1,
    params,
):
    """
    X_t = A_t X_(t-1) + U_t
    U_t ~ N(0, Σ_t)
    """
    σ_ε_sq, σ_Z_sq, φ = params
    return normal_1d(x_t, φ * x_tMinus1, σ_Z_sq)


@jit
def design_func(
    y_t,
    x_t,
    params,
):
    """
    Y_t = B_t X_t + V_t
    V_t ~ N(0, R_t)
    """
    σ_ε_sq, σ_Z_sq, φ = params
    return normal_1d(y_t, x_t, σ_ε_sq)


@pytest.mark.parametrize(
    "σ_ε_sq,σ_Z_sq,φ,G,n_σ,m_sMinus1_given_t,Q_sMinus1_given_t",
    (
        (0.1, 4.0, 1.0, 1000, 3.0, 2.0, 1.5),
        (0.2, 1.7, 0.9, 1500, 5.0, 1.3, 1.9),
    ),
)
def test_one_step_ahead(
    σ_ε_sq,
    σ_Z_sq,
    φ,
    G,
    n_σ,
    m_sMinus1_given_t,
    Q_sMinus1_given_t,
):
    params = np.array([σ_ε_sq, σ_Z_sq, φ])
    σ_Z = np.sqrt(σ_Z_sq)
    x_f = n_σ * σ_Z
    x_0 = -x_f

    grid_to_position = get_grid_to_position(x_0, x_f, G)
    x = grid_to_position(np.arange(G))

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
        grid_to_position,
        p_sMinus1_given_t,
    )

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

    np.testing.assert_allclose(
        p_s_given_t,
        p_s_given_t_ref,
        rtol=0.01,
    )

    return x, p_s_given_t, p_s_given_t_ref


@pytest.mark.parametrize(
    "σ_ε_sq,σ_Z_sq,φ,G,n_σ,m_t_given_tMinus1,Q_t_given_tMinus1,y_t",
    (
        (0.1, 4.0, 1.0, 1000, 3.0, 2.0, 1.5, 1.3),
        (0.2, 1.7, 0.9, 1500, 5.0, 1.3, 1.9, 2.1),
    ),
)
def test_filter_update(
    σ_ε_sq,
    σ_Z_sq,
    φ,
    G,
    n_σ,
    m_t_given_tMinus1,
    Q_t_given_tMinus1,
    y_t,
):
    params = np.array([σ_ε_sq, σ_Z_sq, φ])
    σ_Z = np.sqrt(σ_Z_sq)
    σ_ε = np.sqrt(σ_ε_sq)
    x_f = n_σ * σ_Z
    x_0 = -x_f

    grid_to_position = get_grid_to_position(x_0, x_f, G)
    x = grid_to_position(np.arange(G))

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
        grid_to_position,
        p_t_given_tMinus1,
    )

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

    # np.testing.assert_allclose(
    #     p_t_given_t,
    #     p_t_given_t_ref,
    #     rtol=0.01,
    # )

    return x, p_t_given_t, p_t_given_t_ref


def res_to_df(*args):
    """Convert the above return values to a DataFrame."""
    x, res, ref = args
    return pd.DataFrame(
        {
            "res": res,
            "ref": ref,
        },
        index=x,
    )
