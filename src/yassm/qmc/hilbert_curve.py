from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from logging import getLogger
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numba import jit

from yassm.lib.qmc.hilbert_curve import FLOAT_ORDER, hilbert_sort_inplace
from yassm.lib.qmc.hilbert_curve_full_range import f8_to_u8_func, gen_f8_to_u8_args_func, hilber_sort_inplace_func

if TYPE_CHECKING:
    from collections.abc import Callable
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
        ax: Axes | None = None,
        **kwargs: dict[str, Any],
    ) -> FigAx:
        import matplotlib.pyplot as plt

        array = self.array

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        ax.plot(array[:, 0], array[:, 1], **kwargs)
        ax.set_aspect("equal")
        return fig, ax

    def plot_3d(
        self,
        ax: Axes | None = None,
        **kwargs: dict[str, Any],
    ) -> FigAx:
        import matplotlib.pyplot as plt

        array = self.array

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        ax = fig.add_subplot(projection="3d")
        ax.plot(array[:, 0], array[:, 1], zs=array[:, 2], zdir="z", **kwargs)
        return fig, ax

    def plot(
        self,
        ax: Axes | None = None,
        **kwargs: dict[str, Any],
    ) -> FigAx:
        """Plot the Hilbert curve.

        :param kwargs: kwargs for `ax.plot`
        """
        if self.dimension == 2:
            return self.plot_2d(ax=ax, **kwargs)
        elif self.dimension == 3:
            return self.plot_3d(ax=ax, **kwargs)
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

    def to_float(self) -> HilbertCurveFloat01:
        """Convert the Hilbert curve to float domain."""
        array = self.array / (1 << self.order)
        return HilbertCurveFloat01(array, self.order, check_range=False)


@dataclass
class AbstractHilbertCurveFloat(AbstractHilbertCurve):
    """Any Hilbert curve with float domain."""

    array: npt.NDArray[np.float64]
    order: int = FLOAT_ORDER
    dtype: ClassVar[NumberDType | None] = np.float64

    def to_int(self) -> HilbertCurveInt:
        raise NotImplementedError


@dataclass
class HilbertCurveFloat01(AbstractHilbertCurveFloat):
    """Hilbert curve with float domain.

    The support of each dimension is [0, 1).
    """

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
    ) -> HilbertCurveFloat01:
        """Create a Hilbert curve with the given order.

        Resulted in an array of shape (2**(order * dimension), dimension).
        """
        return HilbertCurveInt.from_order(order, dimension).to_float()

    def to_int(self) -> HilbertCurveInt:
        """Convert the Hilbert curve to integer domain."""
        # maps to [0, 2^order - 1]
        array = np.floor(self.array * (1 << self.order)).astype(np.uint64)
        return HilbertCurveInt(array, self.order, check_range=False)

    def sort(self) -> None:
        """Sort the Hilbert curve inplace."""
        if self.order != FLOAT_ORDER:
            raise ValueError(f"order must be {FLOAT_ORDER}")
        hilbert_sort_inplace(self.array, self.order)

    def sorted(self) -> HilbertCurveFloat01:
        """Sort the Hilbert curve.

        Note that this will be converted to int and sort and convert back to float,
        so precision may be lost.
        """
        h = self.to_int()
        h.sort()
        return h.to_float()


@dataclass
class HilbertCurveFloat(AbstractHilbertCurveFloat):
    """Hilbert curve with float domain.

    :param range_reduction_method: The method to be used to map float to int.
        linear: linear rescaling the range.
        logistic: logistic according to gerber-chopin-2015.
        whiten_cholesky, whiten_ZCA, whiten_ZCA_cor:
            whitening then to uniform using their respective method.

    The support is the full range of float.
    """

    range_reduction_method: str = "linear"

    @cached_property
    def batch_f8_to_u8(
        self,
    ) -> Callable[[npt.NDArray[np.float64], tuple[npt.NDArray[np.float64], ...]], npt.NDArray[np.uint64]]:
        f8_to_u8 = f8_to_u8_func[self.range_reduction_method]

        # "u8[:, ::1](f8[:, ::1])",
        @jit(nopython=True, nogil=True, cache=False)
        def batch_f8_to_u8(
            array: npt.NDArray[np.float64], args: tuple[npt.NDArray[np.float64], ...]
        ) -> npt.NDArray[np.uint64]:
            N, d = array.shape
            out = np.empty((N, d), dtype=np.uint64)
            for i in range(N):
                for j in range(d):
                    out[i, j] = f8_to_u8(array[i], j, args)
            return out

        return batch_f8_to_u8

    def to_int(self) -> HilbertCurveInt:
        """Convert the Hilbert curve to integer domain by `range_reduction_method`."""
        from yassm.qmc.random import whitening_transform

        method = self.range_reduction_method

        if method.startswith("whiten_"):
            whiten_method = "_".join(method.split("_")[1:])
            # to (0, 1) then maps to [0, 2^order - 1]
            array = np.floor(whitening_transform(self.array, method=whiten_method) * (1 << self.order)).astype(
                np.uint64
            )
        # below should always work including whitening but slower
        # as f8_to_u8 is doing matrix multiplication when whitening
        # i.e. it performed a redandant matrix multiplication d times
        else:
            gen_f8_to_u8_args = gen_f8_to_u8_args_func[self.range_reduction_method]
            args = gen_f8_to_u8_args(self.array)
            array = self.batch_f8_to_u8(self.array, args)

        return HilbertCurveInt(array, self.order, check_range=False)

    def sort(self) -> None:
        """Sort the Hilbert curve inplace."""
        if self.order != FLOAT_ORDER:
            raise ValueError(f"order must be {FLOAT_ORDER}")
        hilber_sort_inplace_func[self.range_reduction_method](self.array, self.order)
