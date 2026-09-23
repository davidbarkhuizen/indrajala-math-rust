"""
One Rust function per `ArrayLayer` method, doing the entire computation in a single call instead
of composing it from several separate `Array` operator/ufunc calls in Python. Checked two ways
per function - against real numpy applying the
same formula directly, and against `indrajala_ml.model.array_layer.ArrayLayer` itself (the actual
production reference these functions are meant to replace, method for method), across a random
input sweep.
"""

import random

import numpy as np
import pytest

from indrajala_ml_array import (
    Array,
    layer_accumulate_gradient,
    layer_accumulate_gradient_batch,
    layer_apply_accumulated_gradient,
    layer_forward,
    layer_forward_batch,
    layer_hidden_delta,
    layer_hidden_delta_batch,
    layer_output_delta,
)

from indrajala_ml.model.array_layer import ArrayLayer, sigmoid

SEEDS = range(30)
INPUT_SIZE = 8
HIDDEN_SIZE = 5
NEXT_SIZE = 4
BATCH_SIZE = 6


def _to_numpy(arr):
    if len(arr.shape) == 1:
        return np.array([arr[i] for i in range(arr.shape[0])])
    rows, cols = arr.shape
    return np.array([[arr[r, c] for c in range(cols)] for r in range(rows)])


def _random_vector(rng, n):
    return [rng.uniform(-3.0, 3.0) for _ in range(n)]


def _random_matrix(rng, rows, cols):
    return [_random_vector(rng, cols) for _ in range(rows)]


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_forward_matches_array_layer_forward(seed):
    rng = random.Random(seed)
    w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = _random_vector(rng, HIDDEN_SIZE)
    x_data = _random_vector(rng, INPUT_SIZE)

    layer = ArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    layer.W, layer.b = np.array(w_data), np.array(b_data)
    expected = layer.forward(np.array(x_data))

    actual = layer_forward(Array(w_data), Array(x_data), Array(b_data))
    assert _to_numpy(actual) == pytest.approx(expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_forward_batch_matches_array_layer_forward_batch(seed):
    rng = random.Random(seed)
    w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = _random_vector(rng, HIDDEN_SIZE)
    x_data = _random_matrix(rng, BATCH_SIZE, INPUT_SIZE)

    layer = ArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    layer.W, layer.b = np.array(w_data), np.array(b_data)
    expected = layer.forward_batch(np.array(x_data))

    actual = layer_forward_batch(Array(w_data), Array(x_data), Array(b_data))
    assert _to_numpy(actual) == pytest.approx(expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_output_delta_matches_array_layer_single_and_batch(seed):
    rng = random.Random(seed)

    a_data = [rng.uniform(0.01, 0.99) for _ in range(HIDDEN_SIZE)]
    reference_data = _random_vector(rng, HIDDEN_SIZE)
    layer = ArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    layer.a = np.array(a_data)
    layer.compute_output_delta(np.array(reference_data))
    actual = layer_output_delta(Array(a_data), Array(reference_data))
    assert _to_numpy(actual) == pytest.approx(layer.delta)

    a_batch_data = [[rng.uniform(0.01, 0.99) for _ in range(HIDDEN_SIZE)] for _ in range(BATCH_SIZE)]
    reference_batch_data = _random_matrix(rng, BATCH_SIZE, HIDDEN_SIZE)
    layer.A = np.array(a_batch_data)
    layer.compute_output_delta_batch(np.array(reference_batch_data))
    actual_batch = layer_output_delta(Array(a_batch_data), Array(reference_batch_data))
    assert _to_numpy(actual_batch) == pytest.approx(layer.delta_batch)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_hidden_delta_matches_array_layer_compute_hidden_delta(seed):
    rng = random.Random(seed)
    next_w_data = _random_matrix(rng, NEXT_SIZE, HIDDEN_SIZE)
    next_delta_data = _random_vector(rng, NEXT_SIZE)
    a_data = [rng.uniform(0.01, 0.99) for _ in range(HIDDEN_SIZE)]

    this_layer = ArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    this_layer.a = np.array(a_data)
    next_layer = ArrayLayer(NEXT_SIZE, HIDDEN_SIZE)
    next_layer.W = np.array(next_w_data)
    next_layer.delta = np.array(next_delta_data)
    this_layer.compute_hidden_delta(next_layer)

    actual = layer_hidden_delta(Array(next_w_data), Array(next_delta_data), Array(a_data))
    assert _to_numpy(actual) == pytest.approx(this_layer.delta)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_hidden_delta_batch_matches_array_layer_compute_hidden_delta_batch(seed):
    rng = random.Random(seed)
    next_w_data = _random_matrix(rng, NEXT_SIZE, HIDDEN_SIZE)
    next_delta_batch_data = _random_matrix(rng, BATCH_SIZE, NEXT_SIZE)
    a_batch_data = [[rng.uniform(0.01, 0.99) for _ in range(HIDDEN_SIZE)] for _ in range(BATCH_SIZE)]

    this_layer = ArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    this_layer.A = np.array(a_batch_data)
    next_layer = ArrayLayer(NEXT_SIZE, HIDDEN_SIZE)
    next_layer.W = np.array(next_w_data)
    next_layer.delta_batch = np.array(next_delta_batch_data)
    this_layer.compute_hidden_delta_batch(next_layer)

    actual = layer_hidden_delta_batch(
        Array(next_w_data), Array(next_delta_batch_data), Array(a_batch_data)
    )
    assert _to_numpy(actual) == pytest.approx(this_layer.delta_batch)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_accumulate_gradient_matches_array_layer(seed):
    rng = random.Random(seed)
    delta_data = _random_vector(rng, HIDDEN_SIZE)
    input_activation_data = _random_vector(rng, INPUT_SIZE)
    grad_w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    grad_b_data = _random_vector(rng, HIDDEN_SIZE)

    layer = ArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    layer.delta = np.array(delta_data)
    layer._grad_W = np.array(grad_w_data)
    layer._grad_b = np.array(grad_b_data)
    layer.accumulate_gradient(np.array(input_activation_data))

    new_grad_w, new_grad_b = layer_accumulate_gradient(
        Array(delta_data), Array(input_activation_data), Array(grad_w_data), Array(grad_b_data)
    )
    assert _to_numpy(new_grad_w) == pytest.approx(layer._grad_W)
    assert _to_numpy(new_grad_b) == pytest.approx(layer._grad_b)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_accumulate_gradient_batch_matches_array_layer(seed):
    rng = random.Random(seed)
    delta_batch_data = _random_matrix(rng, BATCH_SIZE, HIDDEN_SIZE)
    input_activation_batch_data = _random_matrix(rng, BATCH_SIZE, INPUT_SIZE)
    grad_w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    grad_b_data = _random_vector(rng, HIDDEN_SIZE)

    layer = ArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    layer.delta_batch = np.array(delta_batch_data)
    layer._grad_W = np.array(grad_w_data)
    layer._grad_b = np.array(grad_b_data)
    layer.accumulate_gradient_batch(np.array(input_activation_batch_data))

    new_grad_w, new_grad_b = layer_accumulate_gradient_batch(
        Array(delta_batch_data),
        Array(input_activation_batch_data),
        Array(grad_w_data),
        Array(grad_b_data),
    )
    assert _to_numpy(new_grad_w) == pytest.approx(layer._grad_W)
    assert _to_numpy(new_grad_b) == pytest.approx(layer._grad_b)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("batch_size", [1, 6])
def test_layer_apply_accumulated_gradient_matches_array_layer(seed, batch_size):
    rng = random.Random(seed)
    w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = _random_vector(rng, HIDDEN_SIZE)
    grad_w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    grad_b_data = _random_vector(rng, HIDDEN_SIZE)
    learning_rate = rng.uniform(0.01, 1.0)

    layer = ArrayLayer(HIDDEN_SIZE, INPUT_SIZE)
    layer.W, layer.b = np.array(w_data), np.array(b_data)
    layer._grad_W, layer._grad_b = np.array(grad_w_data), np.array(grad_b_data)
    layer.apply_accumulated_gradient(learning_rate, batch_size)

    new_w, new_b = layer_apply_accumulated_gradient(
        Array(w_data), Array(b_data), Array(grad_w_data), Array(grad_b_data),
        learning_rate, batch_size,
    )
    assert _to_numpy(new_w) == pytest.approx(layer.W)
    assert _to_numpy(new_b) == pytest.approx(layer.b)


def test_sigmoid_still_matches_the_reference_sigmoid_directly():
    # sanity check that fused.rs's own inline sigmoid (1/(1+e^-z), not a call into ufuncs::exp)
    # matches array_layer.sigmoid's overflow behavior at the same boundary already checked for
    # exp() itself.
    rng = random.Random(0)
    x_data = _random_vector(rng, INPUT_SIZE)
    w_data = [[1000.0] * INPUT_SIZE for _ in range(HIDDEN_SIZE)]
    b_data = [0.0] * HIDDEN_SIZE

    actual = layer_forward(Array(w_data), Array(x_data), Array(b_data))
    expected = sigmoid(np.array(w_data) @ np.array(x_data) + np.array(b_data))
    assert _to_numpy(actual) == pytest.approx(expected)
