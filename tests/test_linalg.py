"""
matmul (as Array's __matmul__, matching every real call site's own `@` syntax), outer, and
sum_axis0 - checked three ways (Rust, numpy, and a
hand-written pure-Python reference loop matching BackpropNode's own per-node sum() formula), a
strictly stronger check than a two-way Rust-vs-numpy comparison alone.
"""

import random

import numpy as np
import pytest

from indrajala_ml_array import Array, outer, sum_axis0


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
