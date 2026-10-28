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

    assert logging.getLogger("yassm").handlers == []


def test_warnings_are_printed_by_default() -> None:
    """Without any logging configuration, Python prints warnings (not info) to stderr, unless a handler is attached.

    yassm warns e.g. when it ignores an argument or falls back to another MLE method.
    """
    import os
    import subprocess
    import sys

    code = """
import logging
import yassm
logger = logging.getLogger("yassm.linear_gaussian")
logger.info("progress")
logger.warning("falling back")
"""
    env = {key: value for key, value in os.environ.items() if key != "YASSMLOGLEVEL"}
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert res.stderr.strip() == "falling back"


def test_default_mle_method_does_not_warn(caplog) -> None:
    """`mle_method` defaults to "", the default method of the model: only an unknown method is replaced, with a warning."""
    from tests import y
    from yassm.linear_gaussian.model.local_level import local_level_model
    from yassm.linear_gaussian.model.local_linear import local_linear_model
    from yassm.linear_gaussian.model.local_ratio import local_ratio_model

    for model, μ_0, Γ_0 in (
        (local_level_model, 0.0, 1e8),
        (local_ratio_model, 0.0, 1e8),
        (local_linear_model, np.zeros(2), 1e8 * np.eye(2)),
    ):
        for mle_method, replaced in (("", False), ("newton", True)):
            caplog.clear()
            with caplog.at_level("WARNING", logger="yassm"):
                model.fit(y.copy(), None, μ_0, Γ_0, mle_method=mle_method, mle_n_iter=1)
            assert ("not supported" in caplog.text or "only implemented" in caplog.text) is replaced


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

    @jit("f8(f8[::1])", nopython=True)
    def g(x):
        return np.linalg.cholesky(np.diag(x)).sum()

    assert as_scipy_objective(g)(np.array([-1.0, 1.0])) == np.inf

    # NumPy words it differently from Numba
    assert as_scipy_objective(lambda x: np.linalg.inv(np.diag(x)).sum())(np.array([0.0, 1.0])) == np.inf


def test_as_scipy_objective_zero_variance() -> None:
    """The scalar kernels, e.g. of the local level model, divide by the variances: a zero one is +inf too.

    It raised ZeroDivisionError, e.g. when L-BFGS-B probed a bound of exactly 0.
    """
    from tests import y
    from yassm.linear_gaussian.model.local_level import local_level_model
    from yassm.util import as_scipy_objective

    objective = as_scipy_objective(local_level_model.neg2_log_likelihood)
    args = (y.reshape(-1, 1), np.zeros(1), np.eye(1))
    assert np.isfinite(objective(np.array([1.0, 1.0]), *args))
    assert objective(np.array([0.0, 1.0]), *args) == np.inf


def test_as_scipy_objective_raises_other_errors() -> None:
    """Only a matrix that cannot be inverted at these parameters is +inf: other errors hold for any parameters."""
    from numba import jit

    from yassm.util import as_scipy_objective

    @jit("f8(f8[::1])", nopython=True)
    def wrong_shape(x):
        return np.linalg.inv(np.ones((2, 3)) * x[0]).sum()

    @jit("f8(f8[::1], f8)", nopython=True)
    def with_data(x, y):
        return np.linalg.solve(np.diag(x), np.full(x.size, y)).sum()

    x = np.array([1.0, 2.0])
    with pytest.raises(np.linalg.LinAlgError, match="square"):
        as_scipy_objective(wrong_shape)(x)
    with pytest.raises(np.linalg.LinAlgError, match="NaN"):
        as_scipy_objective(with_data)(x, np.nan)
    # but the optimizer may try non-finite parameters, e.g. TNC after an infinite objective
    assert as_scipy_objective(with_data)(np.array([1.0, np.nan]), 0.0) == np.inf


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

    # a failed run leaves the fit as it was, including the results derived from the chain
    state = {
        name: getattr(fit, name)
        for name in (
            "gibbs_params_chain",
            "gibbs_X_chain",
            "gibbs_params_chain_with_burn_in",
            "gibbs_X_chain_with_burn_in",
            "m_t_given_T_from_gibbs_samples",
        )
    }
    with pytest.raises(ValueError, match="increase n_sample"):
        fit.run_gibbs_sampling(filter_n_std=1e-6)
    for name, value in state.items():
        assert getattr(fit, name) is value, name

    # filtering does not change what the user asked for: a re-run draws as many samples again
    assert (fit.n_sample, fit.gibbs_burn_in) == (2000, 2000 - n_kept)
    fit.run_gibbs_sampling()
    assert fit.gibbs_params_chain_with_burn_in.shape[0] == 2000
    assert fit.gibbs_params_chain.shape[0] == n_kept


def test_multivariate_observations_are_normalised() -> None:
    """Two independent local levels observed together: the densities are products of the two univariate ones.

    The normalising constants counted a single dimension of the observations, (2π)^(-1/2) instead of (2π)^(-n/2).
    """
    from numba import jit
    from scipy import stats

    from yassm import LinearGaussianModel, LinearGaussianParticleFilterModel, local_level_model
    from yassm.lib.linear_gaussian import prob_normal

    @jit("f8[:, ::1](f8[::1])", nopython=True)
    def identity(params):
        return np.eye(2)

    @jit("f8[:, ::1](f8[::1])", nopython=True)
    def R0(params):
        return np.diag(params[:2])

    @jit("f8[:, ::1](f8[::1])", nopython=True)
    def Σ0(params):
        return np.diag(params[2:])

    @jit("Tuple((f8[::1], f8[:, ::1]))(f8[::1])", nopython=True)
    def X0(params):
        return np.zeros(2), 1e7 * np.eye(2)

    rng = np.random.default_rng(0)
    T = 50
    params = np.array([1.0, 4.0, 1.0, 0.25])
    y = np.column_stack([np.cumsum(rng.normal(size=T)), np.cumsum(0.5 * rng.normal(size=T))]) + rng.normal(size=(T, 2))

    # Kalman filter
    fit = LinearGaussianModel(4, identity, identity, Σ0, R0).fit(y, params, np.zeros(2), 1e7 * np.eye(2))
    expected = sum(local_level_model.fit(y[:, i].copy(), params[[i, 2 + i]], 0.0, 1e7).log_likelihood for i in range(2))
    np.testing.assert_allclose(fit.log_likelihood, expected, rtol=1e-9)

    # weights of the particle filters
    N = 5
    x = rng.normal(size=(N, 2))
    w = np.empty(N)
    model = LinearGaussianParticleFilterModel(4, identity, identity, Σ0, R0, X0=X0)
    model.weights_inplace_bootstrap(w, x, x, y[1], params, 1)
    np.testing.assert_allclose(w, [stats.multivariate_normal(x_i, np.diag(params[:2])).pdf(y[1]) for x_i in x])
    model.weights_inplace_optimal(w, x, x, y[1], params, 1)
    np.testing.assert_allclose(
        w, [stats.multivariate_normal(x_i, np.diag(params[:2] + params[2:])).pdf(y[1]) for x_i in x]
    )

    Σ = np.array([[2.0, 0.5], [0.5, 1.0]])
    np.testing.assert_allclose(
        prob_normal(x, np.ones(2), np.linalg.inv(Σ), False), stats.multivariate_normal(np.ones(2), Σ).pdf(x)
    )


def _exact_log_likelihood(y, A, B, Σ, R, μ_0, Γ_0) -> float:
    """``log p(y_obs[1:] | y_obs[0])`` of a time-invariant model with ``n = 1``, from the joint distribution of ``y_obs``.

    ``X_0 ~ N(μ_0, Γ_0)``, ``X_t = A X_(t-1) + U_t``, ``Y_t = B X_t + V_t``, where ``y_obs`` are the values that are not NaN.
    """
    from scipy import stats

    T = y.size
    means = np.empty(T)
    cov = np.empty((T, T))
    m, P = μ_0, Γ_0
    for t in range(T):
        if t > 0:
            m = A @ m
            P = A @ P @ A.T + Σ
        means[t] = (B @ m)[0]
        # Cov(X_s, X_t) = A^(s - t) Var(X_t) for s >= t
        C = P
        for s in range(t, T):
            cov[s, t] = cov[t, s] = (B @ C @ B.T)[0, 0] + (R[0, 0] if s == t else 0.0)
            C = A @ C
    obs = np.flatnonzero(~np.isnan(y))
    first = obs[0]
    full = stats.multivariate_normal(means[obs], cov[np.ix_(obs, obs)]).logpdf(y[obs])
    return full - stats.norm(means[first], np.sqrt(cov[first, first])).logpdf(y[first])


@pytest.mark.parametrize("model_name", ["local_level_model", "local_ratio_model", "local_linear_model"])
def test_log_likelihood_with_missing_observations(model_name: str) -> None:
    """Missing observations have no term in the log-likelihood, which is conditional on the first one observed.

    It was NaN as soon as an observation was missing, so that ``scipy_minimize`` could not estimate the MLE.
    """
    import yassm

    rng = np.random.default_rng(0)
    T = 60
    y = np.cumsum(rng.normal(size=T)) + rng.normal(scale=2.0, size=T)
    # also at the start: the first observation is y[2]
    y[[0, 1, 20, 21, 45]] = np.nan
    r, q, φ = 4.0, 1.0, 0.9
    if model_name == "local_linear_model":
        A, B, Σ, params = np.array([[2.0, -1.0], [1.0, 0.0]]), np.array([[1.0, 0.0]]), np.diag([q, 0.0]), [r, q]
    elif model_name == "local_ratio_model":
        A, B, Σ, params = np.array([[φ]]), np.eye(1), np.array([[q]]), [r, q, φ]
    else:
        A, B, Σ, params = np.eye(1), np.eye(1), np.array([[q]]), [r, q]
    d = A.shape[0]
    μ_0, Γ_0 = np.zeros(d), 10.0 * np.eye(d)

    model = getattr(yassm, model_name)
    fit = model.fit(y, params, μ_0, Γ_0)
    np.testing.assert_allclose(fit.log_likelihood, _exact_log_likelihood(y, A, B, Σ, np.array([[r]]), μ_0, Γ_0))
    # also at other parameters, in parallel as for the draws of a Gibbs chain, or by scipy_minimize
    paramses = np.array([params, params], dtype=np.float64)
    paramses[1, :2] *= 2.0
    np.testing.assert_allclose(
        fit.log_likelihoods(paramses),
        [fit.log_likelihood, _exact_log_likelihood(y, A, B, 2.0 * Σ, np.array([[2.0 * r]]), μ_0, Γ_0)],
    )


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


@pytest.mark.parametrize("params", [[4.0, 0.25], [4.0, 1e-3]])
def test_mle_hessian_matches_finite_differences(params: list[float]) -> None:
    """The Hessian of the log-likelihood, against central differences with steps relative to each parameter,
    also for parameters of very different scales."""
    pytest.importorskip("numdifftools")
    from yassm import local_level_model

    rng = np.random.default_rng(0)
    σ_ε, σ_Z = np.sqrt(params)
    y = np.cumsum(rng.normal(scale=σ_Z, size=300)) + rng.normal(scale=σ_ε, size=300)
    params = np.array(params)

    def log_likelihood(params: np.ndarray) -> float:
        return local_level_model.fit(y, params, 0.0, 1e8).log_likelihood

    h = 1e-3 * params
    expected = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            e_i = np.eye(2)[i] * h[i]
            e_j = np.eye(2)[j] * h[j]
            expected[i, j] = (
                log_likelihood(params + e_i + e_j)
                - log_likelihood(params + e_i - e_j)
                - log_likelihood(params - e_i + e_j)
                + log_likelihood(params - e_i - e_j)
            ) / (4.0 * h[i] * h[j])
    actual = local_level_model.fit(y, params, 0.0, 1e8).mle_hessian_log_likelihood
    np.testing.assert_allclose(actual, expected, rtol=1e-3)


def test_mle_hessian_at_variance_bound() -> None:
    """An MLE on the variance bound, e.g. a vanishing state noise, has a finite Hessian."""
    pytest.importorskip("numdifftools")
    from yassm import local_level_model
    from yassm.util import VARIANCE_BOUNDS

    y = np.random.default_rng(0).normal(scale=2.0, size=300)
    fit = local_level_model.fit(y, [4.0, VARIANCE_BOUNDS[0]], 0.0, 1e8)
    assert np.all(np.isfinite(fit.mle_hessian_log_likelihood))


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
