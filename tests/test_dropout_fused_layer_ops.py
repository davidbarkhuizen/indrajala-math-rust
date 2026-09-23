"""
layer_dropout_forward/layer_dropout_forward_batch/layer_dropout_hidden_delta/
layer_dropout_hidden_delta_batch, checked against
indrajala_ml.model.dropout_array_layer.DropoutArrayLayer - the actual production reference these
functions replace - the same treatment test_relu_fused_layer_ops.py gives ReLUArrayLayer's own
fused ops.

Unlike every other *_fused_layer_ops.py test module, training=True can't be checked for
bit-identical parity against the numpy-backed reference: this crate's hand-rolled xorshift128+
generator can never reproduce numpy's Mersenne Twister stream (the same RNG-implementation gap
that rules out bit-identical parity for uniform()), and here the mask *is* the mechanism
under test, not incidental to it. So training=False (deterministic, no RNG involved at all) is
checked for exact parity across a random sweep, the same as every other fused op; training=True is
checked structurally instead - shape, {0.0, 1.0}-valued entries, and that layer_dropout_forward's
own returned (a, mask, base_activation) triple is internally consistent with the hand-derived
inverted-dropout formula, plus that layer_dropout_hidden_delta reproduces the correct chain-rule
result when fed a *forced* mask/base_activation pair (sidestepping the RNG entirely).
"""

import random

import numpy as np
import pytest

from indrajala_ml_array import (
    Array,
    layer_dropout_forward,
    layer_dropout_forward_batch,
    layer_dropout_hidden_delta,
    layer_dropout_hidden_delta_batch,
)

from indrajala_ml.model.dropout_array_layer import DropoutArrayLayer

SEEDS = range(30)
INPUT_SIZE = 8
HIDDEN_SIZE = 5
NEXT_SIZE = 4
BATCH_SIZE = 6
DROP_PROBABILITY = 0.4
KEEP_PROBABILITY = 1.0 - DROP_PROBABILITY


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
def test_layer_dropout_forward_at_eval_mode_matches_dropout_array_layer_forward(seed):
    rng = random.Random(seed)
    w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = _random_vector(rng, HIDDEN_SIZE)
    x_data = _random_vector(rng, INPUT_SIZE)

    layer = DropoutArrayLayer(HIDDEN_SIZE, INPUT_SIZE, DROP_PROBABILITY)
    layer.W, layer.b = np.array(w_data), np.array(b_data)
    expected = layer.forward(np.array(x_data))  # training=False by default

    a, mask, base_activation = layer_dropout_forward(
        Array(w_data), Array(x_data), Array(b_data), DROP_PROBABILITY, False
    )
    assert _to_numpy(a) == pytest.approx(expected)
    assert _to_numpy(mask).tolist() == [1.0] * HIDDEN_SIZE
    assert _to_numpy(base_activation) == pytest.approx(expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_dropout_forward_batch_at_eval_mode_matches_dropout_array_layer_forward_batch(seed):
    rng = random.Random(seed)
    w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = _random_vector(rng, HIDDEN_SIZE)
    x_data = _random_matrix(rng, BATCH_SIZE, INPUT_SIZE)

    layer = DropoutArrayLayer(HIDDEN_SIZE, INPUT_SIZE, DROP_PROBABILITY)
    layer.W, layer.b = np.array(w_data), np.array(b_data)
    expected = layer.forward_batch(np.array(x_data))

    a, mask, base_activation = layer_dropout_forward_batch(
        Array(w_data), Array(x_data), Array(b_data), DROP_PROBABILITY, False
    )
    assert _to_numpy(a) == pytest.approx(expected)
    assert _to_numpy(base_activation) == pytest.approx(expected)
    assert mask.shape == (BATCH_SIZE, HIDDEN_SIZE)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_dropout_hidden_delta_at_eval_mode_matches_dropout_array_layer_compute_hidden_delta(seed):
    rng = random.Random(seed)
    next_w_data = _random_matrix(rng, NEXT_SIZE, HIDDEN_SIZE)
    next_delta_data = _random_vector(rng, NEXT_SIZE)
    a_data = [rng.uniform(0.01, 0.99) for _ in range(HIDDEN_SIZE)]

    this_layer = DropoutArrayLayer(HIDDEN_SIZE, INPUT_SIZE, DROP_PROBABILITY)
    this_layer._base_activation = np.array(a_data)
    this_layer._mask = np.ones(HIDDEN_SIZE)
    this_layer._was_training = False
    next_layer = DropoutArrayLayer(NEXT_SIZE, HIDDEN_SIZE, DROP_PROBABILITY)
    next_layer.W = np.array(next_w_data)
    next_layer.delta = np.array(next_delta_data)
    this_layer.compute_hidden_delta(next_layer)

    actual = layer_dropout_hidden_delta(
        Array(next_w_data),
        Array(next_delta_data),
        Array(a_data),
        Array([1.0] * HIDDEN_SIZE),
        KEEP_PROBABILITY,
        False,
    )
    assert _to_numpy(actual) == pytest.approx(this_layer.delta)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_dropout_hidden_delta_batch_at_eval_mode_matches_dropout_array_layer_compute_hidden_delta_batch(seed):
    rng = random.Random(seed)
    next_w_data = _random_matrix(rng, NEXT_SIZE, HIDDEN_SIZE)
    next_delta_batch_data = _random_matrix(rng, BATCH_SIZE, NEXT_SIZE)
    a_batch_data = [[rng.uniform(0.01, 0.99) for _ in range(HIDDEN_SIZE)] for _ in range(BATCH_SIZE)]
    ones_batch = [[1.0] * HIDDEN_SIZE for _ in range(BATCH_SIZE)]

    this_layer = DropoutArrayLayer(HIDDEN_SIZE, INPUT_SIZE, DROP_PROBABILITY)
    this_layer._base_activation_batch = np.array(a_batch_data)
    this_layer._mask_batch = np.ones((BATCH_SIZE, HIDDEN_SIZE))
    this_layer._was_training = False
    next_layer = DropoutArrayLayer(NEXT_SIZE, HIDDEN_SIZE, DROP_PROBABILITY)
    next_layer.W = np.array(next_w_data)
    next_layer.delta_batch = np.array(next_delta_batch_data)
    this_layer.compute_hidden_delta_batch(next_layer)

    actual = layer_dropout_hidden_delta_batch(
        Array(next_w_data),
        Array(next_delta_batch_data),
        Array(a_batch_data),
        Array(ones_batch),
        KEEP_PROBABILITY,
        False,
    )
    assert _to_numpy(actual) == pytest.approx(this_layer.delta_batch)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_dropout_forward_in_training_mode_is_internally_consistent_with_the_hand_derived_formula(seed):
    rng = random.Random(seed)
    w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = _random_vector(rng, HIDDEN_SIZE)
    x_data = _random_vector(rng, INPUT_SIZE)

    a, mask, base_activation = layer_dropout_forward(
        Array(w_data), Array(x_data), Array(b_data), DROP_PROBABILITY, True
    )
    a_values = _to_numpy(a)
    mask_values = _to_numpy(mask)
    base_values = _to_numpy(base_activation)

    for kept, base, actual in zip(mask_values, base_values, a_values):
        assert kept in (0.0, 1.0)
        if kept == 1.0:
            assert actual == pytest.approx(base / KEEP_PROBABILITY)
        else:
            assert actual == 0.0


def test_layer_dropout_forward_batch_in_training_mode_draws_an_independent_mask_per_row():
    rng = random.Random(0)
    w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = _random_vector(rng, HIDDEN_SIZE)
    x_data = _random_matrix(rng, 30, INPUT_SIZE)

    _a, mask, _base = layer_dropout_forward_batch(
        Array(w_data), Array(x_data), Array(b_data), 0.5, True
    )
    rows = [tuple(mask[r, c] for c in range(HIDDEN_SIZE)) for r in range(30)]
    assert len(set(rows)) > 1


def test_layer_dropout_hidden_delta_in_training_mode_uses_the_returned_mask_and_base_activation():
    # sidesteps the RNG entirely by forcing mask/base_activation directly - checks the chain-rule
    # formula, not any particular random outcome
    next_w = Array([[0.8]])
    next_delta = Array([-0.5])
    base_activation = Array([0.7502601055951177])

    kept = layer_dropout_hidden_delta(
        next_w, next_delta, base_activation, Array([1.0]), KEEP_PROBABILITY, True
    )
    dropped = layer_dropout_hidden_delta(
        next_w, next_delta, base_activation, Array([0.0]), KEEP_PROBABILITY, True
    )
    eval_mode = layer_dropout_hidden_delta(
        next_w, next_delta, base_activation, Array([0.0]), KEEP_PROBABILITY, False
    )

    base = 0.7502601055951177
    sigmoid_derivative = base * (1.0 - base)
    expected_kept = (-0.5 * 0.8) * sigmoid_derivative / KEEP_PROBABILITY
    expected_eval = (-0.5 * 0.8) * sigmoid_derivative  # eval-mode ignores mask entirely

    assert kept[0] == pytest.approx(expected_kept)
    assert dropped[0] == 0.0
    assert eval_mode[0] == pytest.approx(expected_eval)


def test_layer_dropout_hidden_delta_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        layer_dropout_hidden_delta(
            Array.zeros((2, 3)), Array.zeros(2), Array.zeros(5), Array.zeros(5), 0.5, True
        )
