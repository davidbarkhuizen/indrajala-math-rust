"""
array_relu/array_relu_mask. Checked directly against numpy's own np.maximum/masking formulas
before ReLUArrayLayer relies on them, the same "prove the primitive against numpy before
building the layer" discipline test_ufuncs_exp.py's own exp check follows.
"""

import random

import numpy as np
import pytest

from indrajala_ml_array import Array, array_relu, array_relu_mask


def _to_flat_list(arr):
    if len(arr.shape) == 1:
        return [arr[i] for i in range(arr.shape[0])]
    rows, cols = arr.shape
    return [arr[r, c] for r in range(rows) for c in range(cols)]


def test_array_relu_matches_numpy_across_a_random_sweep_including_the_zero_boundary():
    rng = random.Random(0)
    values = [rng.uniform(-50.0, 50.0) for _ in range(200)] + [0.0, -0.0, 1e-12, -1e-12]

    expected = np.maximum(0.0, np.array(values))
    actual = _to_flat_list(array_relu(Array(values)))

    for actual_value, expected_value in zip(actual, expected):
        assert actual_value == pytest.approx(float(expected_value), abs=1e-12)


def test_array_relu_preserves_shape_for_2d_arrays():
    matrix = Array([[1.0, -2.0], [-3.0, 4.0]])
    result = array_relu(matrix)
    assert result.shape == (2, 2)
    np_expected = np.maximum(0.0, np.array([[1.0, -2.0], [-3.0, 4.0]]))
    for row in range(2):
        for col in range(2):
            assert result[row, col] == pytest.approx(float(np_expected[row, col]))


def test_array_relu_mask_matches_downstream_times_activation_positive_across_a_random_sweep():
    rng = random.Random(1)
    downstream = [rng.uniform(-10.0, 10.0) for _ in range(50)]
    a = [rng.uniform(-10.0, 10.0) for _ in range(50)]

    expected = np.array(downstream) * (np.array(a) > 0.0)
    actual = _to_flat_list(array_relu_mask(Array(downstream), Array(a)))

    for actual_value, expected_value in zip(actual, expected):
        assert actual_value == pytest.approx(float(expected_value), abs=1e-12)


def test_array_relu_mask_is_zero_exactly_at_the_a_equals_zero_boundary():
    # relu_hidden_delta's own either-branch convention: a == 0 exactly (measure-zero in
    # practice) is treated as zero-derivative, matching np.array(a) > 0.0's strict inequality.
    result = array_relu_mask(Array([5.0]), Array([0.0]))
    assert result[0] == 0.0


def test_array_relu_mask_preserves_shape_for_2d_arrays():
    downstream = Array([[1.0, 2.0], [3.0, 4.0]])
    a = Array([[1.0, -1.0], [-1.0, 1.0]])
    result = array_relu_mask(downstream, a)
    assert result.shape == (2, 2)
    assert result[0, 0] == 1.0
    assert result[0, 1] == 0.0
    assert result[1, 0] == 0.0
    assert result[1, 1] == 4.0


def test_array_relu_mask_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        array_relu_mask(Array.zeros((2, 3)), Array.zeros((3, 2)))
