from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from yassm.lib.qmc.random import halton_sequence

from .hilbert_curve import HilbertCurveFloat

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class HaltonSequence:
    N: int
    dimension: int

    @cached_property
    def array(self) -> npt.NDArray[np.float64]:
        return halton_sequence(self.N, self.dimension, True)

    def to_hilbert_curve(self) -> HilbertCurveFloat:
        return HilbertCurveFloat(self.array, check_range=False)
