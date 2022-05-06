from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg, special

from yassm.lib.qmc.random import correlation_variance_decomposition, halton_sequence

from .hilbert_curve import HilbertCurveFloat

if TYPE_CHECKING:
    import numpy.typing as npt


SQRT2 = np.sqrt(2.0)


# Gaussian #####################################################################


def gaussian_rosenblatt_transform(
    uniform_array: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Transform a Uniform to Gaussian distribution using Rosenblatt transform.

    :param uniform_array: the uniform distribution in (0, 1)^d, shape of (N, d)
    :param cov: the covariance matrix, shape of (d, d)

    U(0, 1)^d to N(0, Σ)
    """
    cov_sqrt = linalg.sqrtm(cov)
    guassian_white = SQRT2 * special.erfinv(2.0 * uniform_array - 1.0)
    return guassian_white @ cov_sqrt


def gaussian_cholesky_decomposition(
    uniform_array: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Transform a Uniform to Gaussian distribution using Cholesky decomposition.

    :param uniform_array: the uniform distribution in (0, 1)^d, shape of (N, d)
    :param cov: the covariance matrix, shape of (d, d)

    U(0, 1)^d to N(0, Σ)
    """
    L = linalg.cholesky(cov)
    guassian_white = SQRT2 * special.erfinv(2.0 * uniform_array - 1.0)
    return guassian_white @ L


def guassian_ZCA_cor(
    uniform_array: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Transform a Uniform to Gaussian distribution using ZCA-cor whitening transformation.

    See eq. 11 from the reference.

    Kessy, Agnan, Alex Lewin, and Korbinian Strimmer. “Optimal Whitening and Decorrelation.” The American Statistician 72, no. 4 (October 2, 2018): 309–14. https://doi.org/10.1080/00031305.2016.1277159.
    """
    σ, P = correlation_variance_decomposition(cov)
    P_sqrt = linalg.sqrtm(P)
    W = P_sqrt * σ
    guassian_white = SQRT2 * special.erfinv(2.0 * uniform_array - 1.0)
    return guassian_white @ W


# QMC ##########################################################################


@dataclass
class HaltonSequence:
    N: int
    dimension: int

    @cached_property
    def array(self) -> npt.NDArray[np.float64]:
        return halton_sequence(self.N, self.dimension, True)

    def to_hilbert_curve(self) -> HilbertCurveFloat:
        return HilbertCurveFloat(self.array, check_range=False)

    def to_gaussian(
        self,
        cov: npt.NDArray[np.float64],
        method: str = "cholesky",
    ) -> HilbertCurveFloat:
        """Transform the Hilbert-sorted Halton-sequence to Gaussian distribution.

        :param cov: the covariance matrix, shape of (d, d)
        :param method: the method to use, "cholesky", "rosenblatt", or "ZCA_cor".
            Default is "cholesky",
            cholesky is faster than rosenblatt,
            ZCA_cor is optimal.
        """
        h = self.to_hilbert_curve()
        h.sort()
        if method == "rosenblatt":
            h.array = gaussian_rosenblatt_transform(h.array, cov)
        elif method == "cholesky":
            h.array = gaussian_cholesky_decomposition(h.array, cov)
        elif method == "ZCA_cor":
            h.array = guassian_ZCA_cor(h.array, cov)
        else:
            raise ValueError("Unknown method: {}".format(method))
        return h
