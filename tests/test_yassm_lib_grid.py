from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numba import jit, vectorize

from yassm.lib import normal_1d
from yassm.lib.grid.one_dimension import get_grid_to_position, one_step_ahead_inplace
from yassm.lib.linear_gaussian.model.local_ratio import one_step_ahead as local_ratio_one_step_ahead


@jit
def transition_func(
    x_t,
    x_tMinus1,
    params,
):
    """
    X_t = A_t X_t-1 + U_t
    """
    σ_ε_sq, σ_Z_sq, φ = params
    return normal_1d(x_t, φ * x_tMinus1, σ_Z_sq)


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
        p_s_given_t_ref / p_s_given_t_ref.sum(),
        rtol=0.01,
    )

    return x, p_s_given_t, p_s_given_t_ref
