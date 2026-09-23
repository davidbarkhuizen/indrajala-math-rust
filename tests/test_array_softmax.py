"""
array_softmax, this crate's row-wise softmax-normalization primitive - checked directly against a
hand-written numpy reference formula (numpy itself has no built-in softmax) before any
softmax array-based class relies on it, the same "prove the primitive against numpy before
building the layer" discipline test_array_relu.py's own array_relu check follows.
"""

import numpy as np
import pytest

from indrajala_ml_array import Array, array_softmax


def _numpy_softmax_1d(z: np.ndarray) -> np.ndarray:
    shifted = z - np.max(z)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def _numpy_softmax_rows(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _to_flat_list(arr):
    if len(arr.shape) == 1:
        return [arr[i] for i in range(arr.shape[0])]
    rows, cols = arr.shape
    return [arr[r, c] for r in range(rows) for c in range(cols)]


def test_array_softmax_matches_numpy_for_a_1d_vector_across_a_random_sweep():
    rng = np.random.default_rng(0)

    for _ in range(100):
        z = rng.uniform(-10.0, 10.0, size=6)
        expected = _numpy_softmax_1d(z)
        actual = _to_flat_list(array_softmax(Array(z.tolist())))
        assert actual == pytest.approx(expected.tolist(), rel=1e-9, abs=1e-12)


def test_array_softmax_sums_to_one_for_a_1d_vector():
    result = array_softmax(Array([1.0, 2.0, 3.0]))
    assert sum(_to_flat_list(result)) == pytest.approx(1.0)


def test_array_softmax_matches_numpy_for_large_magnitude_values_without_overflow():
    # the numerically-adversarial case: large-magnitude z values, confirming the max-shift
    # trick avoids the overflow a naive exp(z)/sum(exp(z)) would hit
    z = np.array([1000.0, 1001.0, 999.0, -1000.0])
    expected = _numpy_softmax_1d(z)
    actual = _to_flat_list(array_softmax(Array(z.tolist())))
    assert actual == pytest.approx(expected.tolist(), rel=1e-9, abs=1e-12)
    assert all(v == v for v in actual)  # no NaN


def test_array_softmax_matches_numpy_row_wise_for_a_2d_matrix_across_a_random_sweep():
    rng = np.random.default_rng(1)

    for _ in range(50):
        z = rng.uniform(-10.0, 10.0, size=(5, 4))
        expected = _numpy_softmax_rows(z)
        actual = array_softmax(Array(z.tolist()))
        assert actual.shape == (5, 4)
        for row in range(5):
            actual_row = [actual[row, col] for col in range(4)]
            assert actual_row == pytest.approx(expected[row].tolist(), rel=1e-9, abs=1e-12)


def test_array_softmax_each_row_sums_to_one_for_a_2d_matrix():
    result = array_softmax(Array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]]))
    for row in range(2):
        assert sum(result[row, col] for col in range(3)) == pytest.approx(1.0)


def test_array_softmax_matches_numpy_row_wise_for_large_magnitude_values_without_overflow():
    z = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]])
    expected = _numpy_softmax_rows(z)
    actual = array_softmax(Array(z.tolist()))
    for row in range(2):
        actual_row = [actual[row, col] for col in range(3)]
        assert actual_row == pytest.approx(expected[row].tolist(), rel=1e-9, abs=1e-12)
