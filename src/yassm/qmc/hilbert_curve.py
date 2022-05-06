from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, ClassVar

import matplotlib.pyplot as plt
import numpy as np

from yassm.lib.qmc.hilbert_curve import FLOAT_ORDER, hilbert_sort_inplace

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt
    from matplotlib.axes._axes import Axes
    from matplotlib.figure import Figure

    FigAx = tuple[Figure, Axes]
    NumberDType = type[np.number[Any]]

logger = getLogger(__name__)


@dataclass
class AbstractHilbertCurve:
    """Abstract class for Hilbert curves."""

    array: npt.NDArray[Any]
    order: int = -1
    check_range: bool = True
    dtype: ClassVar[NumberDType | None] = None

    def __post_init__(self) -> None:
        self.array = np.ascontiguousarray(self.array, dtype=self.dtype)
        if self.dimension >= 64:
            raise ValueError("dimension must be less than 64")
        if not (0 <= self.order < 64):
            raise ValueError("order must be between 0 and 63")

    @classmethod
    def from_order(
        cls,
        order: int,
        dimension: int,
    ) -> AbstractHilbertCurve:
        raise NotImplementedError

    @property
    def N(self) -> int:
        """Returns the no. of points."""
        return self.array.shape[0]

    @property
    def dimension(self) -> int:
        """Returns the dimension of the Hilbert curve."""
        return self.array.shape[1]

    def sort(self) -> None:
        raise NotImplementedError

    def sorted(self) -> AbstractHilbertCurve:
        raise NotImplementedError

    def plot_2d(
        self,
        **kwargs: dict[str, Any],
    ) -> FigAx:
        array = self.array
        fig, ax = plt.subplots()
        ax.plot(array[:, 0], array[:, 1], **kwargs)
        ax.set_aspect("equal")
        return fig, ax

    def plot_3d(
        self,
        **kwargs: dict[str, Any],
    ) -> FigAx:
        array = self.array

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot(array[:, 0], array[:, 1], zs=array[:, 2], zdir="z", **kwargs)
        return fig, ax

    def plot(
        self,
        **kwargs: dict[str, Any],
    ) -> FigAx:
        """Plot the Hilbert curve.

        :param kwargs: kwargs for `ax.plot`
        """
        if self.dimension == 2:
            return self.plot_2d(**kwargs)
        elif self.dimension == 3:
            return self.plot_3d(**kwargs)
        else:
            raise ValueError("dimension must be 2 or 3")


@dataclass
class HilbertCurveInt(AbstractHilbertCurve):
    """Hilbert curve with integer domain.

    The support of each dimension is
        [0, 2**order) == [0, 2**order - 1].
    """

    array: npt.NDArray[np.uint64]
    order: int = -1
    dtype: ClassVar[NumberDType | None] = np.uint64

    def __post_init__(self) -> None:
        auto_order = self.order < 0
        if auto_order:
            # for passing super().__post_init__()
            self.order = 0
        super().__post_init__()
        if auto_order:
            self.order = int(np.ceil(np.log2(self.array.max() + 1)))
            logger.info("Auto choosing Hilbert order: %d", self.order)
        elif self.check_range and self.array.max() + 1 > (1 << self.order):
            raise ValueError("Requires array.max() < 2 ** order")

    @classmethod
    def from_order(
        cls,
        order: int,
        dimension: int,
    ) -> HilbertCurveInt:
        """Create a Hilbert curve with the given order.

        Resulted in an array of shape (2**(order * dimension), dimension).
        """
        x = np.arange(1 << order, dtype=np.uint64)
        xs = np.meshgrid(*(x for _ in range(dimension)))
        array = np.column_stack([x.flatten() for x in xs])
        self = cls(array, order, check_range=False)
        self.sort()
        return self

    def sort(self) -> None:
        """Sort the Hilbert curve inplace."""
        hilbert_sort_inplace(self.array, self.order)

    def to_float(self) -> HilbertCurveFloat:
        """Convert the Hilbert curve to float domain."""
        array = self.array / (1 << self.order)
        return HilbertCurveFloat(array, self.order, check_range=False)


@dataclass
class HilbertCurveFloat(AbstractHilbertCurve):
    """Hilbert curve with float domain.

    The support of each dimension is [0, 1).
    """

    array: npt.NDArray[np.float64]
    order: int = FLOAT_ORDER
    dtype: ClassVar[NumberDType | None] = np.float64

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.check_range:
            if self.array.max() >= 1.0:
                raise ValueError("Requires array.max() < 1")
            elif self.array.min() < 0.0:
                raise ValueError("Requires array.min() >= 0")

    @classmethod
    def from_order(
        cls,
        order: int,
        dimension: int,
    ) -> HilbertCurveFloat:
        """Create a Hilbert curve with the given order.

        Resulted in an array of shape (2**(order * dimension), dimension).
        """
        return HilbertCurveInt.from_order(order, dimension).to_float()

    def to_int(self) -> HilbertCurveInt:
        """Convert the Hilbert curve to integer domain."""
        array = np.round(self.array * (1 << self.order)).astype(np.uint64)
        return HilbertCurveInt(array, self.order, check_range=False)

    def sort(self) -> None:
        """Sort the Hilbert curve inplace."""
        if self.order != FLOAT_ORDER:
            raise ValueError(f"order must be {FLOAT_ORDER}")
        hilbert_sort_inplace(self.array, self.order)

    def sorted(self) -> HilbertCurveFloat:
        """Sort the Hilbert curve.

        Note that this will be converted to int and sort and convert back to float,
        so precision may be lost.
        """
        h = self.to_int()
        h.sort()
        return h.to_float()
