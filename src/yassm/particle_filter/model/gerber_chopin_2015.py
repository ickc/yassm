"""This implements the Gerber-Chopin-2015 model.

α = params[0]

See section 2.9 on reference 2,
section 13.4.2 on reference 3.

Gerber, Mathieu, and Nicolas Chopin. “Sequential Quasi Monte Carlo.” Journal of the Royal Statistical Society: Series B (Statistical Methodology) 77, no. 3 (June 2015): 509–79. https://doi.org/10.1111/rssb.12104.
Owen, Art B., and Peter W. Glynn, eds. Monte Carlo and Quasi-Monte Carlo Methods. Vol. 241. Springer Proceedings in Mathematics & Statistics. Cham: Springer International Publishing, 2018. https://doi.org/10.1007/978-3-319-91436-7.
CHOPIN, NICOLAS, and Omiros Papaspiliopoulos. INTRODUCTION TO SEQUENTIAL MONTE CARLO. S.l.: SPRINGER, 2021.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import jit

from ...lib.linear_gaussian.model.gerber_chopin_2015 import get_ABΣR
from ...linear_gaussian import GeneralLinearGaussianStateSpaceModel
from .. import GeneralLinearGaussianParticleFilterStateSpaceModel

if TYPE_CHECKING:
    import numpy.typing as npt


def get_gerber_chopin_2015_model(
    d: int,
    particle_filter_method: str = "bootstrap",
) -> GeneralLinearGaussianStateSpaceModel:
    (
        A0,
        B0,
        Σ0,
        R0,
    ) = get_ABΣR(d)

    @jit("Tuple((f8[::1], f8[:, ::1]))(f8[::1])", nopython=True, nogil=True, cache=False)
    def X0(params: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return np.zeros(d), np.eye(d)

    return GeneralLinearGaussianParticleFilterStateSpaceModel(
        1,
        A0,
        B0,
        Σ0,
        R0,
        X0=X0,
        param_names=["α"],
        particle_filter_method=particle_filter_method,
    )
