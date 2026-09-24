"""
matmul (as Array's __matmul__, matching every real call site's own `@` syntax), outer, and
sum_axis0 - checked three ways (Rust, numpy, and a
hand-written pure-Python reference loop matching BackpropNode's own per-node sum() formula), a
strictly stronger check than a two-way Rust-vs-numpy comparison alone.
"""

import os
import random
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
    layer_accumulate_gradient,
    layer_accumulate_gradient_batch,
    layer_apply_accumulated_gradient,
    layer_relu_forward_batch,
    layer_sgd_step,
    matmul_threads_for,
    outer,
    set_matmul_threading,
    sum_axis0,
)


def _to_numpy(arr):
    if len(arr.shape) == 1:
        return np.array([arr[i] for i in range(arr.shape[0])])
    rows, cols = arr.shape
    return np.array([[arr[r, c] for c in range(cols)] for r in range(rows)])


def _random_vector(rng, n):
    return [rng.uniform(-3.0, 3.0) for _ in range(n)]


def _random_matrix(rng, rows, cols):
    return [_random_vector(rng, cols) for _ in range(rows)]


def _pure_python_matvec(matrix, vector):
    return [sum(matrix[row][k] * vector[k] for k in range(len(vector))) for row in range(len(matrix))]


def _pure_python_vecmat(vector, matrix):
    cols = len(matrix[0])
    return [
        sum(vector[k] * matrix[k][col] for k in range(len(vector))) for col in range(cols)
    ]


def _pure_python_matmat(a, b):
    rows, inner, cols = len(a), len(b), len(b[0])
    return [
        [sum(a[row][k] * b[k][col] for k in range(inner)) for col in range(cols)]
        for row in range(rows)
    ]


@pytest.mark.parametrize("seed", range(15))
def test_matrix_at_vector_matches_numpy_and_pure_python(seed):
    rng = random.Random(seed)
    w_data = _random_matrix(rng, 4, 6)
    x_data = _random_vector(rng, 6)

    result = Array(w_data) @ Array(x_data)
    expected_numpy = np.array(w_data) @ np.array(x_data)
    expected_python = _pure_python_matvec(w_data, x_data)

    actual = _to_numpy(result)
    assert actual == pytest.approx(expected_numpy)
    assert actual == pytest.approx(expected_python)


@pytest.mark.parametrize("seed", range(15))
def test_vector_at_matrix_matches_numpy_and_pure_python(seed):
    rng = random.Random(seed)
    x_data = _random_vector(rng, 5)
    w_data = _random_matrix(rng, 5, 3)

    result = Array(x_data) @ Array(w_data)
    expected_numpy = np.array(x_data) @ np.array(w_data)
    expected_python = _pure_python_vecmat(x_data, w_data)

    actual = _to_numpy(result)
    assert actual == pytest.approx(expected_numpy)
    assert actual == pytest.approx(expected_python)


@pytest.mark.parametrize("seed", range(15))
def test_matrix_at_matrix_matches_numpy_and_pure_python(seed):
    rng = random.Random(seed)
    a_data = _random_matrix(rng, 3, 4)
    b_data = _random_matrix(rng, 4, 5)

    result = Array(a_data) @ Array(b_data)
    expected_numpy = np.array(a_data) @ np.array(b_data)
    expected_python = np.array(_pure_python_matmat(a_data, b_data))

    actual = _to_numpy(result)
    assert actual == pytest.approx(expected_numpy)
    assert actual == pytest.approx(expected_python)


def test_matmul_rejects_incompatible_shapes():
    with pytest.raises(ValueError):
        Array.zeros((3, 4)) @ Array.zeros((5, 6))


@pytest.mark.parametrize("seed", range(15))
def test_outer_matches_numpy_and_pure_python(seed):
    rng = random.Random(seed)
    a_data = _random_vector(rng, 4)
    b_data = _random_vector(rng, 3)

    result = outer(Array(a_data), Array(b_data))
    expected_numpy = np.outer(np.array(a_data), np.array(b_data))
    expected_python = np.array([[a * b for b in b_data] for a in a_data])

    actual = _to_numpy(result)
    assert actual == pytest.approx(expected_numpy)
    assert actual == pytest.approx(expected_python)


def test_outer_rejects_non_vector_inputs():
    with pytest.raises(ValueError):
        outer(Array.zeros((2, 2)), Array([1.0, 2.0]))


@pytest.mark.parametrize("seed", range(15))
def test_sum_axis0_matches_numpy_and_pure_python(seed):
    rng = random.Random(seed)
    matrix_data = _random_matrix(rng, 5, 4)

    result = sum_axis0(Array(matrix_data))
    expected_numpy = np.array(matrix_data).sum(axis=0)
    expected_python = [
        sum(matrix_data[row][col] for row in range(5)) for col in range(4)
    ]

    actual = _to_numpy(result)
    assert actual == pytest.approx(expected_numpy)
    assert actual == pytest.approx(expected_python)


def test_sum_axis0_rejects_a_1d_array():
    with pytest.raises(ValueError):
        sum_axis0(Array([1.0, 2.0, 3.0]))


def test_accumulate_gradient_batch_formula_matches_numpy():
    # self._grad_b += self.delta_batch.sum(axis=0) - accumulate_gradient_batch's own formula
    rng = random.Random(2)
    delta_batch_data = _random_matrix(rng, 6, 3)

    grad_b = sum_axis0(Array(delta_batch_data))
    expected = np.array(delta_batch_data).sum(axis=0)

    assert _to_numpy(grad_b) == pytest.approx(expected)


def test_accumulate_gradient_formula_matches_numpy():
    # self._grad_W += np.outer(self.delta, input_activation) - accumulate_gradient's own formula
    rng = random.Random(3)
    delta_data = _random_vector(rng, 4)
    input_activation_data = _random_vector(rng, 6)

    grad_w = outer(Array(delta_data), Array(input_activation_data))
    expected = np.outer(np.array(delta_data), np.array(input_activation_data))

    assert _to_numpy(grad_w) == pytest.approx(expected)


_SPECIAL_VALUES = [0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308 / 3, 1e300, -1e-300]


def _vector_with_special_values(rng, n):
    return [rng.choice(_SPECIAL_VALUES) if rng.random() < 0.3 else rng.uniform(-3.0, 3.0) for _ in range(n)]


@pytest.mark.parametrize("seed", range(20))
def test_layer_accumulate_gradient_is_bit_identical_to_grad_w_plus_outer(seed):
    # the one-pass op against the two-pass composition it replaced, compared exactly (not
    # approx): same bits, including signed zeros and subnormals, in delta, x and grad_w
    rng = random.Random(seed)
    m, n = rng.randint(1, 40), rng.randint(1, 70)
    delta = Array(_vector_with_special_values(rng, m))
    x = Array(_vector_with_special_values(rng, n))
    grad_w = Array([_vector_with_special_values(rng, n) for _ in range(m)])
    grad_b = Array(_vector_with_special_values(rng, m))

    new_grad_w, new_grad_b = layer_accumulate_gradient(delta, x, grad_w, grad_b)

    expected_w = (grad_w + outer(delta, x)).tolist()
    assert [[struct.pack("<d", v) for v in row] for row in new_grad_w.tolist()] == [
        [struct.pack("<d", v) for v in row] for row in expected_w
    ]
    assert [struct.pack("<d", v) for v in new_grad_b.tolist()] == [
        struct.pack("<d", v) for v in (grad_b + delta).tolist()
    ]


def test_layer_accumulate_gradient_matches_numpy():
    rng = random.Random(7)
    delta, x = _random_vector(rng, 32), _random_vector(rng, 5408)
    grad_w = _random_matrix(rng, 32, 5408)
    new_grad_w, _ = layer_accumulate_gradient(Array(delta), Array(x), Array(grad_w), Array.zeros(32))
    assert np.array_equal(np.array(new_grad_w.tolist()), np.array(grad_w) + np.outer(delta, x))


@pytest.mark.parametrize(
    "delta_shape, x_shape, grad_w_shape",
    [((3, 1), 4, (3, 4)), (3, (4, 1), (3, 4)), (3, 4, (4, 3)), (3, 4, 4), (3, 4, 12)],
)
def test_layer_accumulate_gradient_rejects_mismatched_shapes(delta_shape, x_shape, grad_w_shape):
    with pytest.raises(ValueError):
        layer_accumulate_gradient(
            Array.zeros(delta_shape), Array.zeros(x_shape), Array.zeros(grad_w_shape), Array.zeros(3)
        )


def _bits(values):
    return [struct.pack("<d", v) for v in values]


def _sgd_step_inputs(rng, m, n):
    delta = Array(_vector_with_special_values(rng, m))
    x = Array(_vector_with_special_values(rng, n))
    w = Array([_vector_with_special_values(rng, n) for _ in range(m)])
    b = Array(_vector_with_special_values(rng, m))
    return w, b, delta, x


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("learning_rate", [0.5, 0.1, 1e-3, 3.0])
def test_layer_sgd_step_is_bit_identical_to_accumulate_into_zeros_then_apply(seed, learning_rate):
    rng = random.Random(seed)
    m, n = rng.randint(1, 40), rng.randint(1, 70)
    w, b, delta, x = _sgd_step_inputs(rng, m, n)

    new_w, new_b = layer_sgd_step(w, b, delta, x, learning_rate)

    grad_w, grad_b = layer_accumulate_gradient(delta, x, Array.zeros((m, n)), Array.zeros(m))
    expected_w, expected_b = layer_apply_accumulated_gradient(w, b, grad_w, grad_b, learning_rate, 1)
    assert [_bits(row) for row in new_w.tolist()] == [_bits(row) for row in expected_w.tolist()]
    assert _bits(new_b.tolist()) == _bits(expected_b.tolist())


def test_layer_sgd_step_inputs_reach_the_signed_zero_case():
    # a -0.0 weight with a -0.0 product stays -0.0 with the 0.0 + (-0.0 - lr * +0.0), but would
    # become +0.0 without it (-0.0 - lr * -0.0). _vector_with_special_values draws -0.0 often
    # enough that the bit-identity test above hits this case, so it would catch the 0.0 + being
    # dropped
    w, b, delta, x = Array([[-0.0, -0.0]]), Array([-0.0]), Array([-0.0]), Array([1.0, 0.0])
    new_w, new_b = layer_sgd_step(w, b, delta, x, 0.5)
    assert _bits(new_w.tolist()[0]) == _bits([-0.0, -0.0])
    assert _bits(new_b.tolist()) == _bits([-0.0])
    assert _bits([-0.0 - 0.5 * (-0.0 * 1.0)]) == _bits([0.0])


@pytest.mark.parametrize(
    "w_shape, b_shape, delta_shape, x_shape",
    [((3, 4), 3, (3, 1), 4), ((3, 4), 3, 3, (4, 1)), ((4, 3), 3, 3, 4), ((3, 4), 4, 3, 4), (12, 3, 3, 4)],
)
def test_layer_sgd_step_rejects_mismatched_shapes(w_shape, b_shape, delta_shape, x_shape):
    with pytest.raises(ValueError):
        layer_sgd_step(Array.zeros(w_shape), Array.zeros(b_shape), Array.zeros(delta_shape), Array.zeros(x_shape), 0.5)


def _fma(a, b, c):
    # a * b + c with one rounding: exact in Fraction, then int / int true division, which CPython
    # rounds correctly
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def _grouped_dot(a, b):
    """linalg.rs's dot_product grouping: lane j accumulates indices j, j+4, ... by FMA, the lanes
    combine as (l0 + l1) + (l2 + l3), and the k % 4 tail continues sequentially by FMA."""
    lanes = [0.0] * 4
    i = 0
    while i + 4 <= len(a):
        for j in range(4):
            lanes[j] = _fma(a[i + j], b[i + j], lanes[j])
        i += 4
    total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
    for t in range(i, len(a)):
        total = _fma(a[t], b[t], total)
    return total


# row counts 1-19 cover every mix of the 8-, 4- and 2-row blocks and a single leftover row;
# widths cover k < 4, k % 4 == 0 and every nonzero k % 4
DOT_SHAPES = [(m, k) for m in range(1, 20) for k in (1, 3, 4, 9, 14, 64)] + [(30, 784), (32, 203)]


@pytest.mark.parametrize("m, k", DOT_SHAPES)
def test_matrix_at_vector_is_the_grouped_dot_product_exactly(m, k):
    rng = np.random.default_rng(m * 1000 + k)
    W = rng.uniform(-1.0, 1.0, size=(m, k)).tolist()
    x = rng.uniform(-1.0, 1.0, size=k).tolist()
    assert (Array(W) @ Array(x)).tolist() == [_grouped_dot(row, x) for row in W]


@pytest.mark.parametrize("m, k", DOT_SHAPES[::3] + [(30, 784)])
def test_forward_batch_rows_are_the_grouped_dot_product_exactly(m, k):
    # X @ W.T (matmul_nt) through layer_relu_forward_batch with b = 0, which returns max(z, 0):
    # every positive z must equal the grouping bit for bit, and every other entry is 0
    rng = np.random.default_rng(m * 1000 + k + 1)
    W = rng.uniform(-1.0, 1.0, size=(m, k)).tolist()
    X = rng.uniform(-1.0, 1.0, size=(3, k)).tolist()
    expected = [[max(_grouped_dot(row, x), 0.0) for row in W] for x in X]
    assert layer_relu_forward_batch(Array(W), Array(X), Array([0.0] * m)).tolist() == expected


# (batch, k, n): batches 1-9 give 0-2 full 4-row blocks of X with 0-3 rows left over; odd n
# leaves a W row outside matmul_nt's 2-row tiles, and n = 1 has no tile at all
NT_TILE_SHAPES = [(batch, k, n) for batch in range(1, 10) for k, n in ((9, 5), (14, 1), (3, 2))] + [
    (9, 64, 17),
    (8, 203, 30),
]


@pytest.mark.parametrize("batch, k, n", NT_TILE_SHAPES)
def test_forward_batch_tiles_are_the_grouped_dot_product_exactly(batch, k, n):
    # every output of matmul_nt's X-row by W-row tiles, and of the rows and columns left over
    # outside them, is dot_product(W[j], X[i]) in its grouping
    rng = np.random.default_rng(batch * 1000 + k * 10 + n)
    W = rng.uniform(-1.0, 1.0, size=(n, k)).tolist()
    X = rng.uniform(-1.0, 1.0, size=(batch, k)).tolist()
    expected = [[max(_grouped_dot(row, x), 0.0) for row in W] for x in X]
    assert layer_relu_forward_batch(Array(W), Array(X), Array([0.0] * n)).tolist() == expected


def _fma_chain(a_row, b, col):
    """matmul's matrix @ matrix output [row, col]: an FMA chain over k increasing from 0.0."""
    total = 0.0
    for k, a_value in enumerate(a_row):
        total = _fma(a_value, b[k][col], total)
    return total


# widths hit each of the kernel's column paths: 16-wide tiles, 4-wide tiles, the scalar tail, and
# mixes of them
TILE_WIDTHS = (1, 3, 4, 5, 15, 16, 17, 20, 21, 35)


# m from 1 to 2 * TILE_ROWS + 1 covers 1-row products, whole row tiles, and a leftover row after
# one or two tiles; at k = 600 a 16 KB row block is 3 rows, so m = 7 leaves a leftover row at the
# end of every block (blocks of 3, 3, 1)
@pytest.mark.parametrize(
    "m, k, n",
    [(m, k, n) for m in range(1, 6) for k in (1, 2, 7) for n in TILE_WIDTHS] + [(7, 600, n) for n in TILE_WIDTHS],
)
def test_matrix_at_matrix_is_the_fma_chain_exactly(m, k, n):
    rng = np.random.default_rng(m * 10000 + k * 100 + n)
    A = rng.uniform(-1.0, 1.0, size=(m, k))
    A[rng.random(A.shape) < 0.2] = 0.0
    B = rng.uniform(-1.0, 1.0, size=(k, n)).tolist()
    expected = [[_fma_chain(a_row, B, col) for col in range(n)] for a_row in A.tolist()]
    assert (Array(A.tolist()) @ Array(B)).tolist() == expected


# (m, k, n) at the sizes the kernel's blocking and threading switch on: the dense layers' batch
# downstream and accumulate (5408 and 784 wide), some over the 8M-flop threading threshold, K large
# enough that a 16 KB row block is a few rows or less than one row, and a tall narrow product
BIG_SHAPES = [
    (8, 32, 5408),
    (32, 32, 5408),
    (32, 128, 5408),
    (30, 512, 784),
    (512, 30, 784),
    (10, 512, 784),
    (5, 3000, 21),
    (300, 9, 35),
    (64, 64, 600),
]


@pytest.mark.parametrize("m, k, n", BIG_SHAPES)
def test_matrix_at_matrix_rows_are_the_vector_at_matrix_product_exactly(m, k, n):
    # the vector @ matrix case runs the same kernel as a one-row product, so every row of a
    # blocked, threaded product must equal it bit for bit
    rng = np.random.default_rng(m * 10000 + k + n)
    A = rng.uniform(-1.0, 1.0, size=(m, k))
    A[rng.random(A.shape) < 0.1] = 0.0
    B = Array(rng.uniform(-1.0, 1.0, size=(k, n)).tolist())
    expected = [(Array(a_row) @ B).tolist() for a_row in A.tolist()]
    assert (Array(A.tolist()) @ B).tolist() == expected


def _fma_chain_row(a, B):
    """_fma_chain for every column of a vector @ matrix product, one pass over B's rows."""
    out = [0.0] * len(B[0])
    for a_value, b_row in zip(a, B):
        out = [_fma(a_value, b_value, total) for b_value, total in zip(b_row, out)]
    return out


# every column path at a few k, and the dense layers' single-example downstream shapes (the conv
# tail's 32 x 5408, 30 x 784, dense MNIST's 10 x 30), which the tests above only compare with the
# matrix @ matrix case, the same kernel
@pytest.mark.parametrize(
    "k, n", [(k, n) for k in (1, 2, 7) for n in TILE_WIDTHS] + [(32, 5408), (30, 784), (10, 30)]
)
def test_vector_at_matrix_is_the_fma_chain_exactly(k, n):
    # pins the reference the test above compares against
    rng = np.random.default_rng(k * 100 + n)
    a = rng.uniform(-1.0, 1.0, size=k)
    a[rng.random(k) < 0.1] = 0.0
    a = a.tolist()
    B = rng.uniform(-1.0, 1.0, size=(k, n)).tolist()
    assert (Array(a) @ Array(B)).tolist() == _fma_chain_row(a, B)


# thread counts that give even and uneven row blocks, and more threads than rows at (5, 3000, 21)
THREAD_COUNTS = [2, 3, 5, 8]
UNTHREADED = (1, 2**62)


@pytest.fixture
def reset_matmul_threading():
    yield
    set_matmul_threading(0, 0)


def _under_each_thread_count(compute):
    # compute() unthreaded, then with every thread count at threshold 1 so even small products
    # split; returns the unthreaded result and each threaded one
    set_matmul_threading(*UNTHREADED)
    unthreaded = compute()
    threaded = {}
    for threads in THREAD_COUNTS:
        set_matmul_threading(threads, 1)
        threaded[threads] = compute()
    return unthreaded, threaded


@pytest.mark.parametrize("m, k, n", BIG_SHAPES)
def test_thread_count_cannot_change_matmul_bits(reset_matmul_threading, m, k, n):
    rng = np.random.default_rng(m * 10000 + k + n)
    A = Array(rng.uniform(-1.0, 1.0, size=(m, k)).tolist())
    B = Array(rng.uniform(-1.0, 1.0, size=(k, n)).tolist())
    unthreaded, threaded = _under_each_thread_count(lambda: (A @ B).tolist())
    for threads, result in threaded.items():
        assert result == unthreaded, threads


@pytest.mark.parametrize("m, k, n", BIG_SHAPES)
def test_thread_count_cannot_change_matmul_nt_bits(reset_matmul_threading, m, k, n):
    # X (m, k) @ W.T for W (n, k), through layer_relu_forward_batch as the grouped-dot test above
    rng = np.random.default_rng(m * 10000 + k + n + 1)
    W = Array(rng.uniform(-1.0, 1.0, size=(n, k)).tolist())
    X = Array(rng.uniform(-1.0, 1.0, size=(m, k)).tolist())
    b = Array(rng.uniform(-1.0, 1.0, size=n).tolist())
    unthreaded, threaded = _under_each_thread_count(lambda: layer_relu_forward_batch(W, X, b).tolist())
    for threads, result in threaded.items():
        assert result == unthreaded, threads


def _accumulate_case(m, k, n):
    # delta_batch (k, m) and X (k, n), so the update is the (m, k) @ (k, n) product delta.T @ X;
    # grad_w has zeros and -0.0 among its values, delta_batch zeros
    rng = np.random.default_rng(m * 10000 + k * 100 + n + 2)
    delta = rng.uniform(-1.0, 1.0, size=(k, m))
    delta[rng.random(delta.shape) < 0.1] = 0.0
    grad_w = rng.uniform(-1.0, 1.0, size=(m, n))
    grad_w[rng.random(grad_w.shape) < 0.1] = 0.0
    grad_w[rng.random(grad_w.shape) < 0.1] = -0.0
    X = rng.uniform(-1.0, 1.0, size=(k, n))
    return Array(delta.tolist()), Array(X.tolist()), Array(grad_w.tolist()), Array([0.0] * m)


# every column path and row tile at short k, and the dense accumulate shapes
@pytest.mark.parametrize(
    "m, k, n", [(m, k, n) for m in range(1, 6) for k in (1, 2, 7) for n in TILE_WIDTHS] + BIG_SHAPES
)
def test_accumulate_gradient_batch_is_the_separate_add_exactly(m, k, n):
    # the add fused into the product's store is the separate grad_w + delta.T @ X bit for bit:
    # the product's chains start from 0.0, and grad_w joins only the finished chain
    delta, X, grad_w, grad_b = _accumulate_case(m, k, n)
    expected = (grad_w + delta.T @ X).tolist()
    new_grad_w, _new_grad_b = layer_accumulate_gradient_batch(delta, X, grad_w, grad_b)
    assert new_grad_w.tolist() == expected


@pytest.mark.parametrize("m, k, n", BIG_SHAPES)
def test_thread_count_cannot_change_accumulate_gradient_batch_bits(reset_matmul_threading, m, k, n):
    delta, X, grad_w, grad_b = _accumulate_case(m, k, n)
    unthreaded, threaded = _under_each_thread_count(
        lambda: layer_accumulate_gradient_batch(delta, X, grad_w, grad_b)[0].tolist()
    )
    for threads, result in threaded.items():
        assert result == unthreaded, threads


def test_accumulate_gradient_batch_rejects_a_mismatched_grad_w():
    delta, X, _grad_w, grad_b = _accumulate_case(3, 2, 5)
    with pytest.raises(ValueError):
        layer_accumulate_gradient_batch(delta, X, Array([[0.0] * 4] * 3), grad_b)


# (input_height, input_width, input_channels, kernel_size, channel_count, stride), batch size:
# MNIST's first conv layer, and a stride-2 multi-channel layer with an odd row count
CONV_CASES = [((28, 28, 1, 3, 32, 1), 4), ((9, 7, 3, 3, 5, 2), 3)]


@pytest.mark.parametrize("shape, n", CONV_CASES)
def test_thread_count_cannot_change_conv_bits(reset_matmul_threading, shape, n):
    # matmul_narrow through all three conv ops: forward (cols @ W.T), downstream (D @ W) and
    # accumulate (D @ cols)
    height, width, channels, kernel_size, channel_count, stride = shape
    g = ConvGeometry(height, width, channels, kernel_size, stride)
    rng = np.random.default_rng(height * 100 + n)
    W = Array(rng.uniform(-1.0, 1.0, size=(channel_count, g.fan_in)).tolist())
    X = Array(rng.uniform(-1.0, 1.0, size=(n, g.input_size)).tolist())
    b = Array(rng.uniform(-1.0, 1.0, size=channel_count).tolist())
    delta = Array(rng.uniform(-1.0, 1.0, size=(n, channel_count * g.positions)).tolist())
    grad_W0 = Array(np.zeros((channel_count, g.fan_in)).tolist())
    grad_b0 = Array(np.zeros(channel_count).tolist())

    def compute():
        A, cols = conv_forward_batch(W, X, b, g)
        dX = conv_downstream_batch(W, delta, g)
        grad_W, grad_b = conv_accumulate_gradient_batch(delta, cols, grad_W0.copy(), grad_b0.copy(), g)
        return [A.tolist(), dX.tolist(), grad_W.tolist(), grad_b.tolist()]

    unthreaded, threaded = _under_each_thread_count(compute)
    for threads, result in threaded.items():
        assert result == unthreaded, threads


# the products the demos thread or used to, (m, k, n): the MNIST conv mini-batch 32's dense tail
# (5.5M flops), dense MNIST 784 -> 30 at batch 512 (12M), MNIST conv accumulate at N = 512 (8
# rows, 24.9M) and the conv mini-batch 512's dense tail (88.6M)
CONV_TAIL_BATCH_32 = (32, 32, 5408)
DENSE_BATCH_512 = (512, 784, 30)
CONV_ACCUMULATE_512 = (8, 512 * 676, 9)
CONV_TAIL_BATCH_512 = (32, 512, 5408)


def test_policy_runs_the_conv_mini_batch_32_tail_on_one_thread(reset_matmul_threading):
    # threading it made the MNIST conv mini-batch 32 epoch 11% slower (stage 0 of the workplan)
    set_matmul_threading(0, 0)
    assert matmul_threads_for(*CONV_TAIL_BATCH_32) == 1
    assert matmul_threads_for(32, 5408, 32) == 1  # its forward, (32, 5408) @ W.T


def test_policy_threads_the_batch_512_products_all_or_nothing(reset_matmul_threading):
    # they gained end to end (or came out even); no 2- or 4-thread middle ground
    # (the count itself is the machine's parallelism, capped at 8)
    set_matmul_threading(0, 0)
    counts = {shape: matmul_threads_for(*shape) for shape in (DENSE_BATCH_512, CONV_ACCUMULATE_512, CONV_TAIL_BATCH_512)}
    assert len(set(counts.values())) == 1, counts
    assert 1 <= counts[CONV_TAIL_BATCH_512] <= 8
    if len(os.sched_getaffinity(0)) >= 2:
        assert counts[CONV_TAIL_BATCH_512] >= 2


def test_policy_override_and_row_cap(reset_matmul_threading):
    set_matmul_threading(3, 1)
    assert matmul_threads_for(*CONV_TAIL_BATCH_32) == 3
    set_matmul_threading(8, 1)
    assert matmul_threads_for(5, 3000, 21) == 5
    set_matmul_threading(*UNTHREADED)
    assert matmul_threads_for(*CONV_TAIL_BATCH_512) == 1
