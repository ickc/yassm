"""The shapes of the outputs of the fits, as annotated with jaxtyping (see yassm._typing)."""

from __future__ import annotations

import numpy as np
import pytest
from jaxtyping import Float64, TypeCheckError, jaxtyped

from tests import set_seed, y
from tests.test_yassm_particle_filter_local_linear import A0, B0, R0, X0, Γ_0, Σ0, μ_0
from yassm._typing import TYPECHECK
from yassm.linear_gaussian.model.local_linear import local_linear_model
from yassm.particle_filter import LinearGaussianParticleFilterModel

T = 50
N = 40
T_FORECAST = 7


def test_typecheck_is_on() -> None:
    """The test suite checks the annotated shapes (set in conftest.py)."""
    assert TYPECHECK


def test_wrong_shape_raises() -> None:
    fit = local_linear_model.fit(y[:T], [1.0, 1.0], μ_0, Γ_0)
    fit._m_t_given_t = fit.m_t_given_t[:, 0]
    with pytest.raises(TypeCheckError):
        _ = fit.filtered_state


def test_linear_gaussian_shapes() -> None:
    """Every output of a fit agrees on T, n, d, p and N."""
    set_seed(0)
    fit = local_linear_model.fit(y[:T], None, μ_0, Γ_0, mle_n_iter=1, n_sample=N, gibbs_burn_in=10)
    with jaxtyped("context"):
        assert isinstance(fit.y, Float64[np.ndarray, "T n"])
        assert isinstance(fit.params, Float64[np.ndarray, " p"])
        assert isinstance(fit.filtered_state, Float64[np.ndarray, "T d"])
        assert isinstance(fit.filtered_state_cov, Float64[np.ndarray, "T d d"])
        assert isinstance(fit.predicted_state, Float64[np.ndarray, "T d"])
        assert isinstance(fit.smoothed_state_cov, Float64[np.ndarray, "T d d"])
        assert isinstance(fit.smoothed_obs, Float64[np.ndarray, "T n"])
        assert isinstance(fit.smoothed_obs_cov_with_obs_cov, Float64[np.ndarray, "T n n"])
        assert isinstance(fit.samples, Float64[np.ndarray, "T N_samples d"])
        assert isinstance(fit.smoothed_obs_from_samples, Float64[np.ndarray, "T n"])
        assert isinstance(fit.gibbs_params_chain, Float64[np.ndarray, "N_gibbs p"])
        assert isinstance(fit.gibbs_X_chain, Float64[np.ndarray, "N_gibbs T d"])
        assert isinstance(fit.smoothed_obs_from_gibbs_samples, Float64[np.ndarray, "T n"])
        for array in fit.forecast_obs_CI(T_FORECAST, 0.05):
            assert isinstance(array, Float64[np.ndarray, "T_forecast n"])
        assert isinstance(fit.log_likelihoods(fit.gibbs_params_chain), Float64[np.ndarray, " N_gibbs"])
    assert fit.y.shape == (T, 1)
    assert fit.samples.shape[1] == N
    assert fit.gibbs_params_chain.shape[0] == N - 10
    assert fit.get_forecast_state(T_FORECAST)[0].shape == (T_FORECAST, 2)


def test_particle_filter_shapes() -> None:
    set_seed(0)
    model = LinearGaussianParticleFilterModel(
        2, A0, B0, Σ0, R0, X0=X0, param_names=local_linear_model.param_names, particle_filter_method="bootstrap"
    )
    fit = model.fit(y[:T], [1.0, 1.0], n_sample=N)
    with jaxtyped("context"):
        assert isinstance(fit.y, Float64[np.ndarray, "T n"])
        assert isinstance(fit.x_t, Float64[np.ndarray, "T N d"])
        assert isinstance(fit.x_t_tilde, Float64[np.ndarray, "T N d"])
        assert isinstance(fit.w_t, Float64[np.ndarray, "T N"])
        assert isinstance(fit.m_t_given_tMinus1, Float64[np.ndarray, "T d"])
        assert isinstance(fit.Q_t_given_t, Float64[np.ndarray, "T d d"])
        assert isinstance(fit.filtered_obs_samples, Float64[np.ndarray, "T N n"])
        assert isinstance(fit.Q_t_given_t_obs, Float64[np.ndarray, "T n n"])
        assert isinstance(fit.get_forecast_state(T_FORECAST), Float64[np.ndarray, "T_forecast N d"])
    assert fit.x_t.shape == (T, N, 2)
