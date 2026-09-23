"""
exp's overflow behavior (does Rust's f64::exp saturate to infinity for large arguments the same
way numpy's np.exp does) - checked with the
same large-z overflow-boundary sweep tests/test_array_layer.py already runs for sigmoid, applied
directly to exp before sigmoid itself is ever built on top of it here.
"""

import random

import numpy as np
import pytest

from indrajala_ml_array import Array, exp


def _to_list(arr):
    return [arr[i] for i in range(arr.shape[0])]


def test_exp_matches_numpy_across_a_random_sweep_including_the_overflow_boundary():
    rng = random.Random(0)
    z_values = [rng.uniform(-50.0, 50.0) for _ in range(200)]
    z_values += [-700.0, -709.0, -710.0, -1000.0, -1e10, 700.0, 709.0, 710.0, 1000.0, 0.0]

    with np.errstate(over="ignore"):
        expected = np.exp(np.array(z_values))

    actual = _to_list(exp(Array(z_values)))

    for actual_value, expected_value in zip(actual, expected):
        # plain pytest.approx (no abs= override), matching test_exp_preserves_shape_for_2d_arrays
        # below: a fixed abs=1e-9 tolerance is meaningless once exp(z) reaches large finite
        # magnitudes (this sweep's own z values run up to the ~709 overflow boundary, giving
        # outputs as large as ~1e307) - float64 itself can't represent differences anywhere near
        # that fine at that scale (one ULP is already >>1e-9), so a fixed-abs comparison would
        # fail on ordinary 1-ULP disagreement between Rust's f64::exp and numpy's own exp, not a
        # real correctness bug. pytest.approx's default combined rel/abs tolerance scales with
        # magnitude and still easily catches a genuine algorithmic error.
        assert actual_value == pytest.approx(float(expected_value))


def test_exp_saturates_to_infinity_for_large_negative_z_matching_sigmoids_own_limit():
    # 1.0 / (1.0 + exp(-z)) for a large negative z should land on the same 0.0 limiting value
    # array_layer.sigmoid's own OverflowError-avoiding formula relies on.
    z = -1000.0
    exp_of_minus_z = exp(Array([-z]))[0]
    assert exp_of_minus_z == float("inf")
    assert 1.0 / (1.0 + exp_of_minus_z) == 0.0


def test_exp_preserves_shape_for_2d_arrays():
    matrix = Array([[0.0, 1.0], [-1.0, 2.0]])
    result = exp(matrix)
    assert result.shape == (2, 2)
    np_expected = np.exp(np.array([[0.0, 1.0], [-1.0, 2.0]]))
    for row in range(2):
        for col in range(2):
            assert result[row, col] == pytest.approx(float(np_expected[row, col]))
