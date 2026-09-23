"""
ConvGeometry and the conv ops (conv_forward_batch, conv_downstream_batch,
conv_accumulate_gradient_batch), plus layer_downstream/layer_downstream_batch. Checked against a
brute-force numpy reference written from the definition of a 'valid' strided convolution -
nested loops over example, output position, channel and kernel offset, no im2col - so the check
is independent of both this crate's im2col formulation and indrajala-ml's numpy ConvArrayLayer.
"""

import itertools

import numpy as np
import pytest

from indrajala_math_rust import (
    Array,
    ConvGeometry,
    conv_accumulate_gradient_batch,
    conv_downstream_batch,
    conv_forward_batch,
    layer_downstream,
    layer_downstream_batch,
)

# (input_height, input_width, input_channels, kernel_size, channel_count, stride): 1/2/3 input
# channels, stride 1/2/3, non-square inputs, a kernel as large as the input, and a stride past
# the kernel that leaves some inputs outside every receptive field
SHAPES = [
    (5, 5, 1, 3, 2, 1),
    (6, 6, 1, 2, 3, 2),
    (5, 7, 3, 3, 2, 1),
    (7, 6, 3, 2, 2, 2),
    (6, 6, 2, 2, 2, 3),
    (4, 4, 2, 4, 3, 1),
    (8, 8, 1, 3, 4, 1),
]
BATCH_SIZE = 3


def _np(arr):
    return np.array(arr.tolist())


def _geometry(shape):
    height, width, channels, kernel_size, _channel_count, stride = shape
    return ConvGeometry(height, width, channels, kernel_size, stride)


def _out_size(size, kernel_size, stride):
    return (size - kernel_size) // stride + 1


def _receptive_field(shape):
    """Every (out_row, out_col, c, kr, kc, in_row, in_col) the convolution reads."""
    height, width, channels, k, _channel_count, s = shape
    for out_row, out_col in itertools.product(range(_out_size(height, k, s)), range(_out_size(width, k, s))):
        for c, kr, kc in itertools.product(range(channels), range(k), range(k)):
            yield out_row, out_col, c, kr, kc, out_row * s + kr, out_col * s + kc


def _reference_forward(W, X, b, shape):
    height, width, channels, k, channel_count, s = shape
    out_h, out_w = _out_size(height, k, s), _out_size(width, k, s)
    planes = X.reshape(len(X), channels, height, width)
    Z = np.zeros((len(X), channel_count, out_h, out_w))
    for n, o in itertools.product(range(len(X)), range(channel_count)):
        Z[n, o] = b[o]
        for out_row, out_col, c, kr, kc, in_row, in_col in _receptive_field(shape):
            Z[n, o, out_row, out_col] += W[o, (c * k + kr) * k + kc] * planes[n, c, in_row, in_col]
    return Z.reshape(len(X), -1)


def _reference_downstream(W, delta, shape):
    height, width, channels, k, channel_count, s = shape
    out_h, out_w = _out_size(height, k, s), _out_size(width, k, s)
    D = delta.reshape(len(delta), channel_count, out_h, out_w)
    dX = np.zeros((len(delta), channels, height, width))
    for n, o in itertools.product(range(len(delta)), range(channel_count)):
        for out_row, out_col, c, kr, kc, in_row, in_col in _receptive_field(shape):
            dX[n, c, in_row, in_col] += W[o, (c * k + kr) * k + kc] * D[n, o, out_row, out_col]
    return dX.reshape(len(delta), -1)


def _reference_gradient(X, delta, shape):
    height, width, channels, k, channel_count, s = shape
    out_h, out_w = _out_size(height, k, s), _out_size(width, k, s)
    planes = X.reshape(len(X), channels, height, width)
    D = delta.reshape(len(X), channel_count, out_h, out_w)
    grad_W = np.zeros((channel_count, channels * k * k))
    for n, o in itertools.product(range(len(X)), range(channel_count)):
        for out_row, out_col, c, kr, kc, in_row, in_col in _receptive_field(shape):
            grad_W[o, (c * k + kr) * k + kc] += D[n, o, out_row, out_col] * planes[n, c, in_row, in_col]
    return grad_W, D.sum(axis=(0, 2, 3))


def _random_case(seed, shape):
    rng = np.random.default_rng(seed)
    height, width, channels, k, channel_count, s = shape
    size = channel_count * _out_size(height, k, s) * _out_size(width, k, s)
    W = rng.uniform(-1.0, 1.0, size=(channel_count, channels * k * k))
    b = rng.uniform(-0.5, 0.5, size=channel_count)
    X = rng.uniform(-1.0, 1.0, size=(BATCH_SIZE, channels * height * width))
    delta = rng.uniform(-1.0, 1.0, size=(BATCH_SIZE, size))
    return W, b, X, delta


@pytest.mark.parametrize("shape", SHAPES)
def test_geometry_derives_the_output_shape(shape):
    height, width, channels, k, _channel_count, s = shape
    g = _geometry(shape)
    assert (g.input_height, g.input_width, g.input_channels, g.kernel_size, g.stride) == (height, width, channels, k, s)
    assert (g.out_height, g.out_width) == (_out_size(height, k, s), _out_size(width, k, s))
    assert g.positions == g.out_height * g.out_width
    assert g.fan_in == channels * k * k
    assert g.input_size == channels * height * width


@pytest.mark.parametrize(
    "arguments",
    [
        (3, 3, 0, 2, 1),  # input_channels
        (3, 3, 1, 0, 1),  # kernel_size
        (3, 3, 1, 2, 0),  # stride
        (3, 5, 1, 4, 1),  # kernel taller than the input
        (5, 3, 1, 4, 1),  # kernel wider than the input
    ],
)
def test_geometry_rejects_invalid_arguments(arguments):
    with pytest.raises(ValueError):
        ConvGeometry(*arguments)


def test_geometry_is_frozen():
    g = ConvGeometry(5, 5, 1, 3, 1)
    with pytest.raises(AttributeError):
        g.stride = 2


@pytest.mark.parametrize("shape", SHAPES)
def test_conv_forward_batch_matches_the_brute_force_definition(shape):
    W, b, X, _delta = _random_case(0, shape)
    Z, A, cols = conv_forward_batch(Array(W.tolist()), Array(X.tolist()), Array(b.tolist()), _geometry(shape))

    expected_Z = _reference_forward(W, X, b, shape)
    np.testing.assert_allclose(_np(Z), expected_Z, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(_np(A), np.maximum(0.0, _np(Z)))

    # cols: row n*P + p in row-major output order, columns in W-row order - a pure copy of the
    # inputs each receptive field reads, so exactly equal
    height, width, channels, k, _channel_count, s = shape
    g = _geometry(shape)
    planes = X.reshape(BATCH_SIZE, channels, height, width)
    expected_cols = np.zeros((BATCH_SIZE * g.positions, g.fan_in))
    for n in range(BATCH_SIZE):
        for out_row, out_col, c, kr, kc, in_row, in_col in _receptive_field(shape):
            expected_cols[n * g.positions + out_row * g.out_width + out_col, (c * k + kr) * k + kc] = planes[
                n, c, in_row, in_col
            ]
    np.testing.assert_array_equal(_np(cols), expected_cols)


@pytest.mark.parametrize("shape", SHAPES)
def test_conv_downstream_batch_matches_the_brute_force_definition(shape):
    W, _b, _X, delta = _random_case(1, shape)
    dX = conv_downstream_batch(Array(W.tolist()), Array(delta.tolist()), _geometry(shape))
    np.testing.assert_allclose(_np(dX), _reference_downstream(W, delta, shape), rtol=1e-12, atol=1e-14)


def test_conv_downstream_batch_gives_unread_inputs_exactly_zero_gradient():
    # 6 wide, k=2, s=3: rows/columns 2 and 5 are outside every receptive field
    shape = (6, 6, 2, 2, 2, 3)
    W, _b, _X, delta = _random_case(2, shape)
    dX = _np(conv_downstream_batch(Array(W.tolist()), Array(delta.tolist()), _geometry(shape)))
    planes = dX.reshape(BATCH_SIZE, 2, 6, 6)
    assert np.all(planes[:, :, [2, 5], :] == 0.0)
    assert np.all(planes[:, :, :, [2, 5]] == 0.0)
    assert np.all(planes[:, :, [0, 1, 3, 4]][:, :, :, [0, 1, 3, 4]] != 0.0)


@pytest.mark.parametrize("shape", SHAPES)
def test_conv_accumulate_gradient_batch_matches_the_brute_force_definition(shape):
    W, b, X, delta = _random_case(3, shape)
    g = _geometry(shape)
    _Z, _A, cols = conv_forward_batch(Array(W.tolist()), Array(X.tolist()), Array(b.tolist()), g)

    rng = np.random.default_rng(4)
    grad_W0 = rng.uniform(-1.0, 1.0, size=W.shape)
    grad_b0 = rng.uniform(-1.0, 1.0, size=b.shape)
    grad_W, grad_b = conv_accumulate_gradient_batch(
        Array(delta.tolist()), cols, Array(grad_W0.tolist()), Array(grad_b0.tolist()), g
    )

    expected_grad_W, expected_grad_b = _reference_gradient(X, delta, shape)
    np.testing.assert_allclose(_np(grad_W), grad_W0 + expected_grad_W, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(_np(grad_b), grad_b0 + expected_grad_b, rtol=1e-12, atol=1e-13)


def test_conv_ops_reject_mismatched_shapes():
    g = ConvGeometry(5, 5, 1, 3, 1)  # fan_in 9, 9 positions, input_size 25
    W, b, X = Array.zeros((2, 9)), Array.zeros(2), Array.zeros((3, 25))
    with pytest.raises(ValueError):
        conv_forward_batch(Array.zeros((2, 8)), X, b, g)  # W columns != fan_in
    with pytest.raises(ValueError):
        conv_forward_batch(W, Array.zeros((3, 24)), b, g)  # X columns != input_size
    with pytest.raises(ValueError):
        conv_forward_batch(W, X, Array.zeros(3), g)  # b length != channel_count
    with pytest.raises(ValueError):
        conv_forward_batch(W, Array.zeros(25), b, g)  # a single example is passed as (1, n)
    with pytest.raises(ValueError):
        conv_downstream_batch(W, Array.zeros((3, 17)), g)  # delta columns != channel_count * P
    with pytest.raises(ValueError):
        conv_accumulate_gradient_batch(Array.zeros((3, 18)), Array.zeros((26, 9)), W, b, g)  # cols rows
    with pytest.raises(ValueError):
        conv_accumulate_gradient_batch(Array.zeros((3, 18)), Array.zeros((27, 9)), W, Array.zeros(3), g)


def test_layer_downstream_matches_numpy():
    rng = np.random.default_rng(5)
    W = rng.uniform(-1.0, 1.0, size=(4, 7))
    delta = rng.uniform(-1.0, 1.0, size=4)
    delta_batch = rng.uniform(-1.0, 1.0, size=(3, 4))
    np.testing.assert_allclose(_np(layer_downstream(Array(W.tolist()), Array(delta.tolist()))), W.T @ delta, rtol=1e-12)
    np.testing.assert_allclose(
        _np(layer_downstream_batch(Array(W.tolist()), Array(delta_batch.tolist()))), delta_batch @ W, rtol=1e-12
    )
    with pytest.raises(ValueError):
        layer_downstream(Array(W.tolist()), Array.zeros(7))
    with pytest.raises(ValueError):
        layer_downstream_batch(Array(W.tolist()), Array.zeros((3, 7)))
