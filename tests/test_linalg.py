"""
matmul (as Array's __matmul__, matching every real call site's own `@` syntax), outer, and
sum_axis0 - checked three ways (Rust, numpy, and a
hand-written pure-Python reference loop matching BackpropNode's own per-node sum() formula), a
strictly stronger check than a two-way Rust-vs-numpy comparison alone.
"""

import random
import struct

import numpy as np
import pytest

from indrajala_math_rust import (
    Array,
    layer_accumulate_gradient,
    layer_accumulate_gradient_batch,
    layer_apply_accumulated_gradient,
    layer_sgd_step,
    outer,
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


# (batch K, output M, input N). matmul_2d threads at K*M*N >= 4M flops and blocks once the
# (K, N) right operand exceeds 256 KB; these cover each of the four combinations, K = 1, M = 1,
# N = 1, and sizes that aren't multiples of 4 (AVX2 lanes) or 64 (the row and k blocks).
_BATCH_GRADIENT_SHAPES = [
    (1, 1, 1),
    (1, 30, 784),
    (32, 1, 784),
    (7, 5, 3),
    (32, 30, 784),  # unthreaded, unblocked (dense production shape at batch 32)
    (10, 500, 1000),  # threaded, unblocked
    (100, 3, 401),  # unthreaded, blocked
    (203, 67, 301),  # threaded, blocked, no multiples of 4 or 64
    (512, 30, 784),  # threaded, blocked (dense production shape at batch 512)
]


@pytest.mark.parametrize("k, m, n", _BATCH_GRADIENT_SHAPES)
def test_layer_accumulate_gradient_batch_is_bit_identical_to_grad_w_plus_transpose_matmul(k, m, n):
    # the transpose-free op against the composition it replaced (a copied delta_batch.T, then @),
    # compared exactly: same summation order per output row, so same bits
    rng = np.random.default_rng(k * 1_000_003 + m * 1009 + n)
    delta_batch = Array(rng.uniform(-3.0, 3.0, (k, m)).tolist())
    x_batch = Array(rng.uniform(-3.0, 3.0, (k, n)).tolist())
    grad_w = Array(rng.uniform(-3.0, 3.0, (m, n)).tolist())
    grad_b = Array(rng.uniform(-3.0, 3.0, m).tolist())

    new_grad_w, new_grad_b = layer_accumulate_gradient_batch(delta_batch, x_batch, grad_w, grad_b)

    expected_w = (grad_w + delta_batch.T @ x_batch).tolist()
    assert [_bits(row) for row in new_grad_w.tolist()] == [_bits(row) for row in expected_w]
    assert _bits(new_grad_b.tolist()) == _bits((grad_b + sum_axis0(delta_batch)).tolist())


@pytest.mark.parametrize("seed", range(20))
def test_layer_accumulate_gradient_batch_is_bit_identical_with_special_values(seed):
    rng = random.Random(seed)
    k, m, n = rng.randint(1, 20), rng.randint(1, 40), rng.randint(1, 70)
    delta_batch = Array([_vector_with_special_values(rng, m) for _ in range(k)])
    x_batch = Array([_vector_with_special_values(rng, n) for _ in range(k)])
    grad_w = Array([_vector_with_special_values(rng, n) for _ in range(m)])

    new_grad_w, _ = layer_accumulate_gradient_batch(delta_batch, x_batch, grad_w, Array.zeros(m))

    expected_w = (grad_w + delta_batch.T @ x_batch).tolist()
    assert [_bits(row) for row in new_grad_w.tolist()] == [_bits(row) for row in expected_w]


def test_layer_accumulate_gradient_batch_matches_numpy():
    rng = np.random.default_rng(11)
    delta_batch, x_batch = rng.uniform(-3.0, 3.0, (32, 30)), rng.uniform(-3.0, 3.0, (32, 784))
    grad_w, grad_b = rng.uniform(-3.0, 3.0, (30, 784)), rng.uniform(-3.0, 3.0, 30)
    new_grad_w, new_grad_b = layer_accumulate_gradient_batch(
        Array(delta_batch.tolist()), Array(x_batch.tolist()), Array(grad_w.tolist()), Array(grad_b.tolist())
    )
    assert np.array(new_grad_w.tolist()) == pytest.approx(grad_w + delta_batch.T @ x_batch)
    assert np.array(new_grad_b.tolist()) == pytest.approx(grad_b + delta_batch.sum(axis=0))


@pytest.mark.parametrize(
    "delta_shape, x_shape, grad_w_shape",
    [((4, 3), (5, 2), (3, 2)), (3, (4, 2), (3, 2)), ((4, 3), 2, (3, 2)), ((4, 3), (4, 2), (2, 3))],
)
def test_layer_accumulate_gradient_batch_rejects_mismatched_shapes(delta_shape, x_shape, grad_w_shape):
    with pytest.raises(ValueError):
        layer_accumulate_gradient_batch(
            Array.zeros(delta_shape), Array.zeros(x_shape), Array.zeros(grad_w_shape), Array.zeros(3)
        )
