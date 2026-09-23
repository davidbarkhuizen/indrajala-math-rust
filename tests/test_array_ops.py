"""
Elementwise + - * /, both broadcasting cases (vector+vector, matrix+row-vector), and scalar
operands - checked against real numpy across a randomized sweep, Rust-vs-numpy since no
pure-Python elementwise-array reference exists to compare a third way against.
"""

import random

import numpy as np
import pytest

from indrajala_ml_array import Array


def _to_numpy(arr):
    if len(arr.shape) == 1:
        return np.array([arr[i] for i in range(arr.shape[0])])
    rows, cols = arr.shape
    return np.array([[arr[r, c] for c in range(cols)] for r in range(rows)])


def _random_vector(rng, n):
    return [rng.uniform(-5.0, 5.0) for _ in range(n)]


def _random_matrix(rng, rows, cols):
    return [_random_vector(rng, cols) for _ in range(rows)]


@pytest.mark.parametrize("seed", range(20))
def test_same_shape_vector_ops_match_numpy(seed):
    rng = random.Random(seed)
    a_data = _random_vector(rng, 7)
    b_data = _random_vector(rng, 7)
    a, b = Array(a_data), Array(b_data)
    np_a, np_b = np.array(a_data), np.array(b_data)

    assert _to_numpy(a + b) == pytest.approx(np_a + np_b)
    assert _to_numpy(a - b) == pytest.approx(np_a - np_b)
    assert _to_numpy(a * b) == pytest.approx(np_a * np_b)
    assert _to_numpy(a / b) == pytest.approx(np_a / np_b)


@pytest.mark.parametrize("seed", range(20))
def test_matrix_plus_row_vector_broadcasts_like_numpy(seed):
    rng = random.Random(seed)
    matrix_data = _random_matrix(rng, 4, 5)
    vector_data = _random_vector(rng, 5)
    matrix, vector = Array(matrix_data), Array(vector_data)
    np_matrix, np_vector = np.array(matrix_data), np.array(vector_data)

    assert _to_numpy(matrix + vector) == pytest.approx(np_matrix + np_vector)
    assert _to_numpy(vector + matrix) == pytest.approx(np_vector + np_matrix)


@pytest.mark.parametrize("seed", range(20))
def test_scalar_operands_match_numpy(seed):
    rng = random.Random(seed)
    data = _random_vector(rng, 6)
    arr = Array(data)
    np_arr = np.array(data)
    learning_rate = 0.5
    batch_size = 8

    assert _to_numpy(learning_rate * arr) == pytest.approx(learning_rate * np_arr)
    assert _to_numpy(arr * learning_rate) == pytest.approx(np_arr * learning_rate)
    assert _to_numpy(arr / batch_size) == pytest.approx(np_arr / batch_size)
    assert _to_numpy(1.0 - arr) == pytest.approx(1.0 - np_arr)


def test_output_delta_formula_matches_numpy():
    # (a - reference) * a * (1 - a) - ArrayLayer.compute_output_delta's own formula
    rng = random.Random(0)
    a_data = [rng.uniform(0.01, 0.99) for _ in range(5)]
    reference_data = [0.0, 1.0, 0.0, 0.0, 0.0]

    a = Array(a_data)
    reference = Array(reference_data)
    delta = (a - reference) * a * (1.0 - a)

    np_a = np.array(a_data)
    np_reference = np.array(reference_data)
    np_delta = (np_a - np_reference) * np_a * (1.0 - np_a)

    assert _to_numpy(delta) == pytest.approx(np_delta)


def test_iadd_mutates_in_place_and_matches_numpy():
    a = Array([1.0, 2.0, 3.0])
    b = Array([10.0, 20.0, 30.0])
    a += b
    assert _to_numpy(a) == pytest.approx(np.array([11.0, 22.0, 33.0]))


def test_isub_mutates_in_place_and_matches_numpy():
    a = Array([10.0, 20.0, 30.0])
    b = Array([1.0, 2.0, 3.0])
    a -= b
    assert _to_numpy(a) == pytest.approx(np.array([9.0, 18.0, 27.0]))


def test_apply_accumulated_gradient_formula_matches_numpy():
    # W -= learning_rate * grad_W / batch_size - ArrayLayer.apply_accumulated_gradient's formula
    rng = random.Random(1)
    w_data = _random_matrix(rng, 3, 4)
    grad_data = _random_matrix(rng, 3, 4)
    learning_rate = 0.5
    batch_size = 4

    w = Array(w_data)
    grad_w = Array(grad_data)
    w -= learning_rate * grad_w / batch_size

    np_w = np.array(w_data)
    np_grad_w = np.array(grad_data)
    np_w -= learning_rate * np_grad_w / batch_size

    assert _to_numpy(w) == pytest.approx(np_w)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        Array([1.0, 2.0, 3.0]) + Array([1.0, 2.0])
