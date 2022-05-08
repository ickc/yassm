from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg, special

from ..lib.qmc.random import correlation_variance_decomposition, halton_sequence
from .hilbert_curve import HilbertCurveFloat, HilbertCurveFloat01

if TYPE_CHECKING:
    import numpy.typing as npt

SQRT2 = np.sqrt(2.0)
SQRT2_INV = np.power(2.0, -0.5)
logger = getLogger(__name__)


# Gaussian #####################################################################


def gaussian_transform(
    uniform_array: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64] | None = None,
    method: str = "cholesky",
) -> npt.NDArray[np.float64]:
    """Transform a Uniform to Gaussian distribution.

    :param uniform_array: the uniform distribution in (0, 1)^d, shape of (N, d)
    :param cov: the covariance matrix, shape of (d, d). If None, assume standard Gaussian, i.e. cov = np.eye(d).
    :param method:
        cholesky: Cholesky decomposition, fastest.
        ZCA_cor: ZCA-cor whitening transformation. See eq. 11 from the reference.
        rosenblatt, ZCA: Rosenblatt transform, also called ZCA in the context of Gaussian.

    U(0, 1)^d to N(0, Σ)

    Kessy, Agnan, Alex Lewin, and Korbinian Strimmer. “Optimal Whitening and Decorrelation.” The American Statistician 72, no. 4 (October 2, 2018): 309–14. https://doi.org/10.1080/00031305.2016.1277159.
    """
    # to white gaussian
    guassian_white = SQRT2 * special.erfinv(2.0 * uniform_array - 1.0)

    if cov is None:
        return guassian_white
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
        return guassian_white @ W_inv_T


def whitening_transform(
    gassian_array: npt.NDArray[np.float64],
    mean: npt.NDArray[np.float64] | None = None,
    cov: npt.NDArray[np.float64] | None = None,
    method: str = "cholesky",
) -> npt.NDArray[np.float64]:
    """Whiten a Gaussian distribution to uniform distribution.

    :param gassian_array: the gaussian distribution in N(0, Σ), shape of (N, d)
    :param mean: the mean of the gaussian distribution, shape of (d,). If None, use sample mean.
    :param cov: the covariance matrix, shape of (d, d). If None, use sample covariance.
    :param method:
        ZCA: ZCA whitening
        cholesky: Cholesky decomposition
        ZCA_cor: ZCA-cor whitening transformation

    These are inverse of gaussian_transform with corresponding method,
    except for cholesky because of the cholesky-inv ordering
    """
    if mean is None:
        mean = gassian_array.mean(axis=0)
    if cov is None:
        cov = np.cov(gassian_array, rowvar=False)

    if method == "cholesky":
        # avoid LinAlgError: not positive definite
        cov_inv = linalg.pinv(cov)
        # force symmetric
        cov_inv = 0.5 * (cov_inv + cov_inv.T)
        # minimum eigenvalue
        eigen_min = linalg.eigvalsh(cov_inv)[0]
        # force eigen_min >= 0
        if eigen_min < 0.0:
            logger.warning(
                "cov_inv is not positive definite with min eigenvalue λ=%f. "
                "Forcing it to be positive semi-definite by adding λΙ.",
                eigen_min,
            )
            cov_inv += eigen_min * np.eye(cov_inv.shape[0])

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
    G = (gassian_array - mean) @ W_T
    # to uniform
    return 0.5 + 0.5 * special.erf(G * SQRT2_INV)


# QMC ##########################################################################


@dataclass
class HaltonSequence:
    N: int
    dimension: int

    @cached_property
    def array(self) -> npt.NDArray[np.float64]:
        return halton_sequence(self.N, self.dimension, True)

    def to_hilbert_curve(self) -> HilbertCurveFloat01:
        return HilbertCurveFloat01(self.array, check_range=False)

    def to_gaussian(
        self,
        cov: npt.NDArray[np.float64],
        method: str = "cholesky",
        range_reduction_method: str = "linear",
    ) -> HilbertCurveFloat:
        """Transform the Hilbert-sorted Halton-sequence to Gaussian distribution.

        :param cov: the covariance matrix, shape of (d, d)
        :param method: the method to use, "cholesky", "rosenblatt", or "ZCA_cor".
            Default is "cholesky",
            cholesky is faster than rosenblatt,
            ZCA_cor is optimal.
        :param range_reduction_method: The method to be used to map float to int.
            linear: linear rescaling the range.
            logistic: logistic according to gerber-chopin-2015.
            whiten_cholesky, whiten_ZCA, whiten_ZCA_cor:
                whitening then to uniform using their respective method.
        """
        h = self.to_hilbert_curve()
        h.sort()
        logger.info("Transforming to Gaussian with method %s", method)
        array = gaussian_transform(h.array, cov, method=method)
        return HilbertCurveFloat(array, order=h.order, check_range=False, range_reduction_method=range_reduction_method)
