from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float64
from scipy import linalg, special

from .._typing import typechecked
from ..lib.qmc.random import correlation_variance_decomposition, halton_sequence
from .hilbert_curve import HilbertCurveFloat, HilbertCurveFloat01

if TYPE_CHECKING:
    pass

SQRT2 = np.sqrt(2.0)
SQRT2_INV = np.power(2.0, -0.5)
EPS = np.finfo(np.float64).eps
logger = getLogger(__name__)


# Gaussian #####################################################################


@typechecked
def gaussian_transform(
    uniform_array: Float64[np.ndarray, "N d"],
    cov: Float64[np.ndarray, "d d"] | None = None,
    method: str = "cholesky",
) -> Float64[np.ndarray, "N d"]:
    """Transform a Uniform to Gaussian distribution.

    Parameters
    ----------
    uniform_array
        the uniform distribution in (0, 1)^d, shape of (N, d)
    cov
        the covariance matrix, shape of (d, d). If None, assume standard Gaussian, i.e. cov = np.eye(d).
    method
        - ``cholesky``: Cholesky decomposition, fastest.
        - ``ZCA_cor``: ZCA-cor whitening transformation. See eq. 11 from the reference.
        - ``rosenblatt``, ``ZCA``: Rosenblatt transform, also called ZCA in the context of Gaussian.

    Notes
    -----
    U(0, 1)^d to N(0, Σ)

    Kessy, Agnan, Alex Lewin, and Korbinian Strimmer. “Optimal Whitening and Decorrelation.” The American Statistician 72, no. 4 (October 2, 2018): 309–14. https://doi.org/10.1080/00031305.2016.1277159.
    """
    # to white gaussian
    gaussian_white = SQRT2 * special.erfinv(2.0 * uniform_array - 1.0)

    if cov is None:
        return gaussian_white
    else:
        if method == "cholesky":
            # W_inv_T = U = L.T
            W_inv_T = linalg.cholesky(cov)
        elif method == "ZCA_cor":
            σ, P = correlation_variance_decomposition(cov)
            P_sqrt = linalg.sqrtm(P)
            W_inv_T = P_sqrt * σ
        elif method in ("rosenblatt", "ZCA"):
            W_inv_T = linalg.sqrtm(cov)
        else:
            raise ValueError(f"Unknown method: {method}")
        # inverse whitening
        return gaussian_white @ W_inv_T


@typechecked
def whitening_transform(
    gaussian_array: Float64[np.ndarray, "N d"],
    mean: Float64[np.ndarray, " d"] | None = None,
    cov: Float64[np.ndarray, "d d"] | None = None,
    method: str = "cholesky",
) -> Float64[np.ndarray, "N d"]:
    """Whiten a Gaussian distribution to uniform distribution.

    Parameters
    ----------
    gaussian_array
        the gaussian distribution in N(0, Σ), shape of (N, d)
    mean
        the mean of the gaussian distribution, shape of (d,). If None, use sample mean.
    cov
        the covariance matrix, shape of (d, d). If None, use sample covariance.
    method
        - ``ZCA``: ZCA whitening
        - ``cholesky``: Cholesky decomposition
        - ``ZCA_cor``: ZCA-cor whitening transformation

    Notes
    -----
    These are inverse of gaussian_transform with corresponding method,
    except for cholesky because of the cholesky-inv ordering
    """
    if mean is None:
        mean = gaussian_array.mean(axis=0)
    if cov is None:
        cov = np.cov(gaussian_array, rowvar=False)

    if method == "cholesky":
        # avoid LinAlgError: not positive definite
        cov_inv = linalg.pinv(cov)
        # force symmetric
        cov_inv = 0.5 * (cov_inv + cov_inv.T)
        # force positive definiteness, e.g. for singular cov:
        # shift the spectrum so that its minimum is at least a tiny positive number
        d = cov_inv.shape[0]
        eigen = linalg.eigvalsh(cov_inv)
        eigen_floor = d * EPS * max(eigen[-1], 1.0)
        if eigen[0] < eigen_floor:
            logger.warning(
                "cov_inv is not positive definite with min eigenvalue λ=%g. Shifting its spectrum by %g.",
                eigen[0],
                eigen_floor - eigen[0],
            )
            cov_inv += (eigen_floor - eigen[0]) * np.eye(d)

        # W_T = L = U.T
        W_T = linalg.cholesky(cov_inv, lower=True)
    elif method == "ZCA":
        W_T = linalg.pinv(linalg.sqrtm(cov))
    elif method == "ZCA_cor":
        σ, P = correlation_variance_decomposition(cov)
        P_sqrt_inv = linalg.pinv(linalg.sqrtm(P))
        W_T = P_sqrt_inv / σ.reshape(-1, 1)
    else:
        raise ValueError(f"Unknown method: {method}")

    # whitening
    G = (gaussian_array - mean) @ W_T
    # to uniform
    return 0.5 + 0.5 * special.erf(G * SQRT2_INV)


# QMC ##########################################################################


@dataclass
class HaltonSequence:
    N: int
    dimension: int

    @cached_property
    @typechecked
    def array(self) -> Float64[np.ndarray, "N d"]:
        return halton_sequence(self.N, self.dimension, True)

    def to_hilbert_curve(self) -> HilbertCurveFloat01:
        return HilbertCurveFloat01(self.array, check_range=False)

    def to_gaussian(
        self,
        cov: Float64[np.ndarray, "d d"],
        method: str = "cholesky",
        range_reduction_method: str = "linear",
    ) -> HilbertCurveFloat:
        """Transform the Hilbert-sorted Halton-sequence to Gaussian distribution.

        Parameters
        ----------
        cov
            the covariance matrix, shape of (d, d)
        method
            the method to use, ``"cholesky"`` (default), ``"rosenblatt"``, or ``"ZCA_cor"``:
            cholesky is faster than rosenblatt, ZCA_cor is optimal.
        range_reduction_method
            The method to be used to map float to int:

            - ``linear``: linear rescaling the range.
            - ``logistic``: logistic according to Gerber & Chopin (2015).
            - ``whiten_cholesky``, ``whiten_ZCA``, ``whiten_ZCA_cor``:
              whitening then to uniform using their respective method.
        """
        h = self.to_hilbert_curve()
        h.sort()
        logger.info("Transforming to Gaussian with method %s", method)
        array = gaussian_transform(h.array, cov, method=method)
        return HilbertCurveFloat(array, order=h.order, check_range=False, range_reduction_method=range_reduction_method)
