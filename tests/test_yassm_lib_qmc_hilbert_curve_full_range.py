from __future__ import annotations

import numpy as np
import pytest

from yassm.lib.qmc.hilbert_curve import FLOAT_ORDER
from yassm.lib.qmc.hilbert_curve_full_range import (
    f8_to_u8_func,
    gen_f8_to_u8_args_func,
    gen_f8_to_u8_args_linear,
)
from yassm.qmc.hilbert_curve import HilbertCurveFloat

U8_MAX = (1 << FLOAT_ORDER) - 1


def test_f8_to_u8_linear_range() -> None:
    """The extremes must map to the extremes of [0, 2^FLOAT_ORDER - 1]."""
    data = np.array([[-3.0, 1.0], [0.5, 2.0], [7.0, 10.0]])
    args = gen_f8_to_u8_args_linear(data)
    f8_to_u8 = f8_to_u8_func["linear"]
    for j in range(data.shape[1]):
        values = [f8_to_u8(row, j, args) for row in data]
        assert values[0] == 0
        assert values[-1] == U8_MAX
        assert values == sorted(values)


@pytest.mark.parametrize("method", ("linear", "logistic"))
def test_full_range_sort_agrees_with_int_sort(method) -> None:
    """Hilbert-sorting floats equals mapping them to uint64 first and sorting there."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2000, 3))
    # an exact maximum per axis exercises the special case in f8_to_u8_linear
    data[0] = data.max(axis=0)
    args = gen_f8_to_u8_args_func[method](data)
    f8_to_u8 = f8_to_u8_func[method]

    h = HilbertCurveFloat(data.copy(), range_reduction_method=method, check_range=False)
    h.sort()
    sorted_then_mapped = np.array([[f8_to_u8(row, j, args) for j in range(3)] for row in h.array], dtype=np.uint64)

    h_int = HilbertCurveFloat(data.copy(), range_reduction_method=method, check_range=False).to_int()
    h_int.sort()
    np.testing.assert_array_equal(sorted_then_mapped, h_int.array)
