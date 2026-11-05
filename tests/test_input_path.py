"""
The input path (the NES plan's D13): Array.from_buffer copies a float64 buffer at memory speed,
with the values Array(x.tolist()) gives; one_hot_planes builds the channel-major one-hot input
from a byte a cell, with the values of numpy's comparison against each channel.
"""

import math

import numpy as np
import pytest

from indrajala_math_rust import Array, one_hot_planes


def _bits(values):
    return [v.hex() if isinstance(v, float) else [x.hex() for x in v] for v in values]


@pytest.mark.parametrize("shape", [(1,), (7,), (1, 5), (3, 4), (256, 6656)])
def test_from_buffer_gives_the_values_of_the_list_constructor(shape):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(shape) * 10.0 ** rng.integers(-300, 300, size=shape)
    assert _bits(Array.from_buffer(x).tolist()) == _bits(Array(x.tolist()).tolist())
    assert Array.from_buffer(x).shape == shape


def test_from_buffer_keeps_signed_zeros_infinities_and_nan():
    x = np.array([[0.0, -0.0, math.inf], [-math.inf, math.nan, 5e-324]])
    values = Array.from_buffer(x).tolist()
    assert [v.hex() for v in values[0]] == ["0x0.0p+0", "-0x0.0p+0", "inf"]
    assert values[1][0] == -math.inf and math.isnan(values[1][1]) and values[1][2] == 5e-324


def test_from_buffer_reads_bytes_of_doubles_too():
    x = np.arange(6.0)
    assert Array.from_buffer(memoryview(x.tobytes()).cast("d")).tolist() == x.tolist()


@pytest.mark.parametrize(
    "data, error",
    [
        (np.zeros((3, 4), dtype=np.float32), BufferError),
        (np.zeros((3, 4), dtype=np.int64), BufferError),
        (np.zeros((3, 8))[:, ::2], ValueError),
        (np.zeros((2, 3, 4)), ValueError),
        (np.zeros((0, 4)), ValueError),
        ([1.0, 2.0], TypeError),
    ],
    ids=["float32", "int64", "strided", "3d", "zero-rows", "list"],
)
def test_from_buffer_refuses_rather_than_converts(data, error):
    with pytest.raises(error):
        Array.from_buffer(data)


def _reference(ids, channels):
    n, cells = ids.shape
    planes = ids[:, None, :] == np.arange(channels)[None, :, None]
    return planes.astype(np.float64).reshape(n, channels * cells)


@pytest.mark.parametrize("n, cells, channels", [(1, 1, 1), (2, 3, 4), (5, 208, 32), (256, 208, 64), (3, 7, 256)])
def test_one_hot_planes_is_numpys_comparison_with_each_channel(n, cells, channels):
    ids = np.random.default_rng(2).integers(0, channels, size=(n, cells), dtype=np.uint8)
    planes = one_hot_planes(ids.tobytes(), cells, channels)
    assert planes.shape == (n, channels * cells)
    assert planes.tolist() == _reference(ids, channels).tolist()


@pytest.mark.parametrize(
    "ids, cells, channels",
    [(b"\x00\x04", 2, 4), (b"\x00\x01\x02", 2, 4), (b"", 2, 4), (b"\x00", 0, 4), (b"\x00", 1, 0)],
    ids=["id-past-channels", "partial-example", "empty", "zero-cells", "zero-channels"],
)
def test_one_hot_planes_refuses_bad_input(ids, cells, channels):
    with pytest.raises(ValueError):
        one_hot_planes(ids, cells, channels)
