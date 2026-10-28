from __future__ import annotations

import importlib

import numpy as np
import pytest

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


def test_import_does_not_require_optional_dependencies() -> None:
    """Importing every module must only need the core dependencies (numba, numpy, scipy)."""
    import subprocess
    import sys

    code = """
import pkgutil, importlib, sys
import yassm
for m in pkgutil.walk_packages(yassm.__path__, "yassm."):
    importlib.import_module(m.name)
optional = ("pandas", "matplotlib", "seaborn", "numdifftools", "statsmodels")
print(",".join(m for m in optional if m in sys.modules))
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert res.stdout.strip() == ""


def test_import_is_quiet() -> None:
    """A library should not configure logging on import."""
    import logging

    import yassm  # noqa: F401

    assert all(isinstance(h, logging.NullHandler) for h in logging.getLogger("yassm").handlers)


@pytest.mark.parametrize(
    ("module", "old", "new"),
    [
        ("yassm.linear_gaussian", "GeneralLinearGaussianStateSpaceModel", "LinearGaussianModel"),
        ("yassm.linear_gaussian", "GeneralLinearGaussianStateSpaceModelFit", "LinearGaussianFit"),
        ("yassm.linear_gaussian", "GenericLinearGaussianStateSpaceModel", "BaseLinearGaussianModel"),
        ("yassm.grid.one_dimension", "General1DStateSpaceModel", "GridFilterModel"),
        ("yassm.grid.one_dimension", "General1DStateSpaceModelFit", "GridFilterFit"),
        ("yassm.particle_filter", "GenericParticleFilterStateSpaceModel", "BaseParticleFilterModel"),
        ("yassm.particle_filter", "GenericParticleFilterStateSpaceModelFit", "ParticleFilterFit"),
        (
            "yassm.particle_filter",
            "GeneralLinearGaussianParticleFilterStateSpaceModel",
            "LinearGaussianParticleFilterModel",
        ),
        (
            "yassm.particle_filter",
            "GeneralNonLinearBootstrapParticleFilterStateSpaceModel",
            "NonLinearParticleFilterModel",
        ),
        (
            "yassm.particle_filter",
            "GeneralNonLinearBootstrapParticleFitlerStateSpaceModel",
            "NonLinearParticleFilterModel",
        ),
        ("yassm.lib.qmc.hilbert_curve_full_range", "hilber_sort_inplace_func", "hilbert_sort_inplace_func"),
    ],
)
def test_deprecated_aliases(module: str, old: str, new: str) -> None:
    mod = importlib.import_module(module)
    with pytest.warns(DeprecationWarning, match=old):
        assert getattr(mod, old) is getattr(mod, new)


def test_unknown_attribute_raises() -> None:
    import yassm.particle_filter

    with pytest.raises(AttributeError, match="no attribute"):
        yassm.particle_filter.NoSuchThing  # noqa: B018


def test_fit_accepts_readonly_input() -> None:
    """e.g. DataFrame.values is read-only under pandas' Copy-on-Write."""
    from tests import y
    from yassm.linear_gaussian.model.local_level import local_level_model

    y_ro = y.copy()
    y_ro.flags.writeable = False
    fit = local_level_model.fit(y_ro, [1.0, 1.0], 0.0, 1e8)
    np.testing.assert_allclose(fit.log_likelihood, local_level_model.fit(y.copy(), [1.0, 1.0], 0.0, 1e8).log_likelihood)


def test_as_scipy_objective() -> None:
    """scipy.optimize may pass a non-contiguous x to the objective."""
    from numba import jit

    from yassm.util import as_scipy_objective

    @jit("f8(f8[::1], f8)", nopython=True)
    def f(x, a):
        return x.sum() + a

    x = np.arange(10.0)[::2]
    assert not x.flags.c_contiguous
    assert as_scipy_objective(f)(x, 1.0) == 21.0


def test_fit_plot_renders() -> None:
    """The legend labels must be valid mathtext; this only fails when the figure is drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tests import y
    from yassm.linear_gaussian.model.local_level import local_level_model

    fit = local_level_model.fit(y, [1.0, 1.0], 0.0, 1e8)
    fig, _ = fit.plot(T_forecast=10)
    fig.canvas.draw()
    plt.close(fig)


def test_fit_plot_given_axes() -> None:
    """The fit is drawn on the given axes and their figure, even if another figure is current."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tests import y
    from yassm.linear_gaussian.model.local_level import local_level_model

    fit = local_level_model.fit(y, [1.0, 1.0], 0.0, 1e8)
    fig, ax = plt.subplots()
    other = plt.figure()
    assert fit.plot(axes=[ax]) == (fig, [ax])
    assert ax.lines
    plt.close(fig)
    plt.close(other)


def test_as_scipy_objective_singular() -> None:
    """A singular matrix at the boundary of the parameter space is +inf, not an error."""
    from numba import jit

    from yassm.util import as_scipy_objective

    @jit("f8(f8[::1])", nopython=True)
    def f(x):
        return np.linalg.inv(np.diag(x)).sum()

    assert as_scipy_objective(f)(np.array([0.0, 1.0])) == np.inf


def test_run_gibbs_sampling_filter() -> None:
    """Filtering keeps the post-burn-in chain length, drops low-likelihood samples and refreshes derived results."""
    from tests import set_seed, y
    from yassm.linear_gaussian.model.local_level import local_level_model

    set_seed(20220501)
    n_kept = 200
    fit = local_level_model.fit(y.copy(), None, 0.0, np.inf, mle_n_iter=100, n_sample=2000, gibbs_burn_in=2000 - n_kept)
    assert fit.gibbs_params_chain_with_burn_in.shape[0] == 2000
    m_unfiltered = fit.m_t_given_T_from_gibbs_samples

    n_std = 1.0
    fit.run_gibbs_sampling(filter_n_std=n_std)
    assert fit.gibbs_params_chain.shape[0] == n_kept
    assert np.all(fit.log_likelihoods(fit.gibbs_params_chain) > fit.log_likelihood - 0.5 * n_std**2)
    assert fit.m_t_given_T_from_gibbs_samples is not m_unfiltered

    chain = fit.gibbs_params_chain
    with pytest.raises(ValueError, match="increase n_sample"):
        fit.run_gibbs_sampling(filter_n_std=1e-6)
    assert fit.gibbs_params_chain is chain
    assert fit.n_sample == fit.gibbs_burn_in + n_kept


def test_top_level_exports() -> None:
    import yassm

    for name in yassm.__all__:
        assert getattr(yassm, name) is not None
    assert yassm.LinearGaussianModel is importlib.import_module("yassm.linear_gaussian").LinearGaussianModel
    assert set(yassm.__all__) <= set(dir(yassm))
    with pytest.raises(AttributeError, match="no attribute"):
        yassm.NoSuchThing  # noqa: B018


@pytest.mark.parametrize("model_name", ["local_level_model", "local_ratio_model", "local_linear_model"])
def test_posterior_normal_approximation(model_name: str) -> None:
    """The Hessian at the MLE works for the built-in models and gives a positive definite covariance."""
    pytest.importorskip("numdifftools")
    import yassm

    rng = np.random.default_rng(42)
    x = np.cumsum(rng.normal(size=300))
    d = 1
    if model_name == "local_linear_model":
        x = np.cumsum(x) / 10.0
        d = 2
    y = x + rng.normal(scale=2.0, size=x.size)

    model = getattr(yassm, model_name)
    n_params = len(model.param_names)
    fit = model.fit(
        y,
        None,
        np.zeros(d),
        1e8 * np.identity(d),
        mle_method="scipy_minimize",
        mle_n_iter=1,
        mle_kwargs={"bounds": [(1e-8, None)] * 2 + [(None, None)] * (n_params - 2)},
    )
    cov = fit.posterior_normal_approximation_cov
    assert cov.shape == (n_params, n_params)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-6)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)


def test_ar2_conditional_mle_is_ols() -> None:
    """The conditional MLE of an AR(2) is ordinary least squares, with σ² the mean squared residual."""
    from scipy.optimize import minimize

    from yassm.lib.linear_gaussian.model.autoregressive_2 import neg_log_likelihood

    rng = np.random.default_rng(0)
    y = np.zeros(500)
    for t in range(2, y.size):
        y[t] = 1.0 + 0.5 * y[t - 1] - 0.3 * y[t - 2] + rng.normal()
    X = np.column_stack((np.ones(y.size - 2), y[1:-1], y[:-2]))
    β, ssr, *_ = np.linalg.lstsq(X, y[2:])
    expected = np.array([ssr[0] / (y.size - 2), *β])

    res = minimize(neg_log_likelihood, [1.0, 0.0, 0.0, 0.0], args=(y,), bounds=[(1e-8, None)] + [(None, None)] * 3)
    np.testing.assert_allclose(res.x, expected, rtol=1e-4)


def test_ma1_mle_default_bounds() -> None:
    """With the default bounds, scipy.optimize reaches statsmodels' MLE instead of stopping at σ² = 0."""
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.arima.model import ARIMA

    from yassm import moving_average_1_model

    # a seed for which a lower bound of exactly 0 made L-BFGS-B stop early
    rng = np.random.default_rng(10)
    e = rng.normal(size=401)
    y = e[1:] - 0.8 * e[:-1]
    μ_0, Γ_0 = np.zeros(2), np.identity(2)  # ignored: X0 is the stationary distribution
    fit = moving_average_1_model.fit(y, [y.var(), 0.0], μ_0, Γ_0, mle_n_iter=1)

    θ_sm, σ_sq_sm = ARIMA(y, order=(0, 0, 1), trend="n").fit().params
    ll_sm = -0.5 * moving_average_1_model.neg2_log_likelihood(np.array([σ_sq_sm, θ_sm]), y.reshape(-1, 1), μ_0, Γ_0)
    assert fit.log_likelihood >= ll_sm - 1e-6


def test_all_matches_exports() -> None:
    import yassm

    assert set(yassm.__all__) == {"__version__", "setup_logging", *yassm._EXPORTS}
