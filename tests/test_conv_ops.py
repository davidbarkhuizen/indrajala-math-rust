"""
ConvGeometry and the conv ops (conv_forward_batch, conv_downstream_batch,
conv_accumulate_gradient_batch), plus layer_downstream/layer_downstream_batch. Checked against a
brute-force numpy reference written from the definition of a 'valid' strided convolution -
nested loops over example, output position, channel and kernel offset, no im2col - so the check
is independent of both this crate's im2col formulation and indrajala-ml's numpy ConvArrayLayer.
"""

import itertools
import struct
from fractions import Fraction

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
    matmul_threads_for,
    set_matmul_threading,
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
    A, cols = conv_forward_batch(Array(W.tolist()), Array(X.tolist()), Array(b.tolist()), _geometry(shape))

    expected_A = np.maximum(0.0, _reference_forward(W, X, b, shape))
    np.testing.assert_allclose(_np(A), expected_A, rtol=1e-12, atol=1e-14)

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


# (shape, N) beyond SHAPES x (1, BATCH_SIZE) for the forward's matmul_narrow kernel, which it
# runs one example at a time on one thread: output channel counts that hit each of its paths
# (16-wide blocks, 4-wide blocks, the scalar tail, and mixes), a 28x28 input at N = 256, fan_in
# 800 x 48 channels (many 16-wide blocks), and 13x13x8 x 32 at N = 32. The last two batch sizes
# are over the threading threshold for a whole-batch product, as the forward's was when it
# ran over the whole batch
FORWARD_EXTRA_CASES = [
    ((6, 6, 1, 3, 1, 1), 2),
    ((6, 6, 2, 3, 5, 1), 2),
    ((6, 6, 2, 3, 16, 1), 2),
    ((6, 6, 2, 3, 21, 1), 2),
    ((6, 6, 2, 3, 35, 1), 2),
    ((28, 28, 1, 3, 8, 1), 1),
    ((28, 28, 1, 3, 8, 1), 256),
    ((8, 8, 32, 5, 48, 1), 2),
    ((13, 13, 8, 3, 32, 1), 32),
]


@pytest.mark.parametrize(
    "shape, n", [(shape, n) for shape in SHAPES for n in (1, BATCH_SIZE)] + FORWARD_EXTRA_CASES
)
def test_conv_forward_batch_is_relu_of_the_matmul_scatter_exactly(shape, n):
    # A is relu(Z) with the ReLU applied during the scatter, and no Z is kept. Rebuild that from
    # its parts - the crate's own matmul (Array @) on the returned cols, the b add, the
    # channel-major scatter, then np.maximum - and require the same bits. The op computes cols @
    # W.T with matmul_narrow, so this is also its exact check against matmul
    rng = np.random.default_rng(5)
    height, width, channels, k, channel_count, _s = shape
    W = rng.uniform(-1.0, 1.0, size=(channel_count, channels * k * k))
    b = rng.uniform(-0.5, 0.5, size=channel_count)
    X = rng.uniform(-1.0, 1.0, size=(n, channels * height * width))
    X[rng.random(X.shape) < 0.1] = 0.0
    g = _geometry(shape)
    A, cols = conv_forward_batch(Array(W.tolist()), Array(X.tolist()), Array(b.tolist()), g)

    by_position = _np(cols @ Array(W.T.tolist()))  # (N*P, O), the same matmul call
    Z = (by_position + b).reshape(n, g.positions, len(b)).transpose(0, 2, 1).reshape(n, -1)
    assert _np(A).tolist() == np.maximum(0.0, Z).tolist()


@pytest.mark.parametrize("shape", SHAPES)
def test_a_vector_operand_is_one_example_with_the_same_bits(shape):
    # every op given one example as a 1D vector returns exactly the (1, size) call's result,
    # flattened, with its per-example outputs as vectors; cols stays (P, C*k*k)
    W, b, X, delta = _random_case(7, shape)
    W, b, g = Array(W.tolist()), Array(b.tolist()), _geometry(shape)
    x, d = X[0].tolist(), delta[0].tolist()

    A_row, cols_row = conv_forward_batch(W, Array([x]), b, g)
    A_vec, cols_vec = conv_forward_batch(W, Array(x), b, g)
    assert A_vec.shape == (A_row.shape[1],)
    assert A_vec.tolist() == A_row.tolist()[0]
    assert cols_vec.shape == cols_row.shape == (g.positions, g.fan_in)
    assert cols_vec.tolist() == cols_row.tolist()

    dX_row = conv_downstream_batch(W, Array([d]), g)
    dX_vec = conv_downstream_batch(W, Array(d), g)
    assert dX_vec.shape == (g.input_size,)
    assert dX_vec.tolist() == dX_row.tolist()[0]

    grad_W0, grad_b0 = Array(np.full(W.shape, 0.25).tolist()), Array(np.full(len(b.tolist()), -0.5).tolist())
    row = conv_accumulate_gradient_batch(Array([d]), cols_row, grad_W0, grad_b0, g)
    vec = conv_accumulate_gradient_batch(Array(d), cols_vec, grad_W0, grad_b0, g)
    assert [r.tolist() for r in vec] == [r.tolist() for r in row]


# (shape, N) beyond SHAPES x (1, BATCH_SIZE) for the backward ops' matmul_narrow calls, whose
# output rows are fan_in wide: 9 (4-wide blocks and a tail), 12 (4-wide blocks only), 18 (a
# 16-wide block and a tail) and 72 (16- and 4-wide blocks); fan_in 800 (many 16-wide blocks); a
# 28x28 input at N = 256 and 13x13x8 x 32 at N = 32 (downstream and accumulate over the 8M-flop
# threading threshold)
BACKWARD_EXTRA_CASES = [
    ((6, 6, 1, 3, 3, 1), 2),
    ((6, 6, 3, 2, 5, 1), 2),
    ((6, 6, 2, 3, 4, 1), 2),
    ((13, 13, 8, 3, 8, 1), 3),
    ((8, 8, 32, 5, 6, 1), 2),
    ((28, 28, 1, 3, 8, 1), 256),
    ((13, 13, 8, 3, 32, 1), 32),
]


THREADED_CASES = [((28, 28, 1, 3, 8, 1), 256), ((13, 13, 8, 3, 32, 1), 32)]


def test_the_threaded_cases_are_over_the_threading_threshold():
    # the backward ops' matmul_narrow products at these cases split across threads at the default
    # threshold (8 threads allowed on any machine), so a threshold change can't silently drop them.
    # The forward runs per example on one thread; its whole-batch product is checked too, as the
    # size its cases were chosen for
    set_matmul_threading(8, 0)
    try:
        for shape, n in THREADED_CASES:
            assert (shape, n) in FORWARD_EXTRA_CASES and (shape, n) in BACKWARD_EXTRA_CASES
            g = _geometry(shape)
            rows, channel_count = n * g.positions, shape[4]
            assert matmul_threads_for(rows, g.fan_in, channel_count) == 8  # forward, whole batch
            assert matmul_threads_for(rows, channel_count, g.fan_in) == 8  # downstream
            assert matmul_threads_for(channel_count, rows, g.fan_in) == 8  # accumulate
    finally:
        set_matmul_threading(0, 0)

def _backward_case(seed, shape, n):
    rng = np.random.default_rng(seed)
    height, width, channels, k, channel_count, _s = shape
    g = _geometry(shape)
    W = rng.uniform(-1.0, 1.0, size=(channel_count, g.fan_in))
    X = rng.uniform(-1.0, 1.0, size=(n, g.input_size))
    delta = rng.uniform(-1.0, 1.0, size=(n, channel_count * g.positions))
    delta[rng.random(delta.shape) < 0.3] = 0.0  # what a ReLU mask leaves
    return W, X, delta, g


@pytest.mark.parametrize(
    "shape, n", [(shape, n) for shape in SHAPES for n in (1, BATCH_SIZE)] + BACKWARD_EXTRA_CASES
)
def test_conv_downstream_batch_is_the_matmul_col2im_exactly(shape, n):
    # rebuild dX from its parts: the crate's own matmul (Array @) on the deltas regrouped to
    # (N*P, O), then col2im as a sequential scatter-add in the op's order (kernel offset outside
    # output position). The op computes D @ W with matmul_narrow, so this is its exact check
    W, _X, delta, g = _backward_case(8, shape, n)
    o, p, k, fan_in = len(W), g.positions, g.kernel_size, g.fan_in
    by_position = delta.reshape(n, o, p).transpose(0, 2, 1).reshape(n * p, o)
    dcols = _np(Array(by_position.tolist()) @ Array(W.tolist())).tolist()

    expected = [[0.0] * g.input_size for _ in range(n)]
    for example in range(n):
        out = expected[example]
        for c, kr, kc in itertools.product(range(g.input_channels), range(k), range(k)):
            column = (c * k + kr) * k + kc
            for out_row, out_col in itertools.product(range(g.out_height), range(g.out_width)):
                row, col = out_row * g.stride + kr, out_col * g.stride + kc
                out[(c * g.input_height + row) * g.input_width + col] += dcols[example * p + out_row * g.out_width + out_col][column]

    dX = conv_downstream_batch(Array(W.tolist()), Array(delta.tolist()), g)
    assert dX.tolist() == expected


@pytest.mark.parametrize(
    "shape, n", [(shape, n) for shape in SHAPES for n in (1, BATCH_SIZE)] + BACKWARD_EXTRA_CASES
)
def test_conv_accumulate_gradient_batch_is_the_matmul_update_exactly(shape, n):
    # grad_W: grad_W0 + (the crate's own matmul of the deltas regrouped to (O, N*P) with cols),
    # one add per element. The op computes D @ cols with matmul_narrow, so this is its exact check
    W, X, delta, g = _backward_case(9, shape, n)
    o, p = len(W), g.positions
    b = np.zeros(o)
    _A, cols = conv_forward_batch(Array(W.tolist()), Array(X.tolist()), Array(b.tolist()), g)
    rng = np.random.default_rng(10)
    grad_W0 = rng.uniform(-1.0, 1.0, size=W.shape)
    by_channel = delta.reshape(n, o, p).transpose(1, 0, 2).reshape(o, n * p)
    expected = grad_W0 + _np(Array(by_channel.tolist()) @ cols)

    grad_W, _grad_b = conv_accumulate_gradient_batch(
        Array(delta.tolist()), cols, Array(grad_W0.tolist()), Array(b.tolist()), g
    )
    assert grad_W.tolist() == expected.tolist()


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
    _A, cols = conv_forward_batch(Array(W.tolist()), Array(X.tolist()), Array(b.tolist()), g)

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
        conv_forward_batch(W, Array.zeros(24), b, g)  # a single-example vector of the wrong length
    with pytest.raises(ValueError):
        conv_downstream_batch(W, Array.zeros(17), g)
    with pytest.raises(ValueError):
        conv_accumulate_gradient_batch(Array.zeros(18), Array.zeros((18, 9)), W, b, g)  # one example: P = 9 cols rows
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
        layer_downstream(Array(W.tolist()), Array.zeros((4, 1)))
    with pytest.raises(ValueError):
        layer_downstream_batch(Array(W.tolist()), Array.zeros((3, 7)))


def _fma(a, b, c):
    # a * b + c rounded once: Fraction arithmetic is exact and float(Fraction) rounds correctly
    # (Python 3.10 has no math.fma). Exact zero sums don't arise from the uniform inputs below,
    # so signed-zero rules don't need handling here
    return float(Fraction(a) * Fraction(b) + Fraction(c))


@pytest.mark.parametrize("m, n", [(1, 1), (1, 9), (9, 1), (4, 7), (10, 30), (30, 13), (37, 301)])
def test_layer_downstream_is_a_sequential_fma_chain(m, n):
    # layer_downstream computes delta @ W as out[j] = fma(delta[k], W[k, j], out[j]) over k in
    # increasing order, from 0.0: the scalar path's definition. Matching it bit for bit here (on
    # an AVX2 machine) shows the AVX2 path gives the same bits, so results are machine-independent
    rng = np.random.default_rng(m * 1000 + n)
    W = rng.uniform(-1.0, 1.0, size=(m, n)).tolist()
    delta = rng.uniform(-1.0, 1.0, size=m).tolist()
    expected = [0.0] * n
    for k in range(m):
        expected = [_fma(delta[k], W[k][j], expected[j]) for j in range(n)]

    actual = layer_downstream(Array(W), Array(delta)).tolist()

    assert [struct.pack("<d", v) for v in actual] == [struct.pack("<d", v) for v in expected]
