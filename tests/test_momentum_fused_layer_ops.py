"""
`layer_momentum_apply_accumulated_gradient` is one fused Rust call for the whole momentum update
rule, checked against
`indrajala_ml.model.momentum_array_layer.MomentumArrayLayer` - the actual production reference
this function replaces - the same treatment `test_adam_fused_layer_ops.py`/
`test_l2_fused_layer_ops.py` give their own fused ops.
"""

import random

import numpy as np
import pytest

from indrajala_ml_array import Array, layer_momentum_apply_accumulated_gradient

from indrajala_ml.model.momentum_array_layer import MomentumArrayLayer

SEEDS = range(30)
INPUT_SIZE = 8
HIDDEN_SIZE = 5
MOMENTUM = 0.5


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
@pytest.mark.parametrize("batch_size", [1, 6])
def test_layer_momentum_apply_accumulated_gradient_matches_momentum_array_layer_at_step_one(seed, batch_size):
    rng = random.Random(seed)
    w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = _random_vector(rng, HIDDEN_SIZE)
    grad_w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    grad_b_data = _random_vector(rng, HIDDEN_SIZE)
    learning_rate = rng.uniform(0.001, 1.0)

    layer = MomentumArrayLayer(HIDDEN_SIZE, INPUT_SIZE, MOMENTUM)
    layer.W, layer.b = np.array(w_data), np.array(b_data)
    layer._grad_W, layer._grad_b = np.array(grad_w_data), np.array(grad_b_data)
    layer.apply_accumulated_gradient(learning_rate, batch_size)

    new_w, new_b, new_prev_delta_w, new_prev_delta_b = layer_momentum_apply_accumulated_gradient(
        Array(w_data),
        Array(b_data),
        Array(grad_w_data),
        Array(grad_b_data),
        Array.zeros((HIDDEN_SIZE, INPUT_SIZE)),
        Array.zeros(HIDDEN_SIZE),
        MOMENTUM,
        learning_rate,
        batch_size,
    )
    assert _to_numpy(new_w) == pytest.approx(layer.W)
    assert _to_numpy(new_b) == pytest.approx(layer.b)
    assert _to_numpy(new_prev_delta_w) == pytest.approx(layer._prev_delta_W)
    assert _to_numpy(new_prev_delta_b) == pytest.approx(layer._prev_delta_b)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_momentum_apply_accumulated_gradient_matches_across_several_steps(seed):
    # the previous-delta state only actually exercises its accumulation logic across repeated
    # steps, unlike a stateless update where a single comparison would do
    rng = random.Random(seed)
    layer = MomentumArrayLayer(HIDDEN_SIZE, INPUT_SIZE, MOMENTUM)
    layer.W = np.array(_random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE))
    layer.b = np.array(_random_vector(rng, HIDDEN_SIZE))
    learning_rate = rng.uniform(0.001, 1.0)

    prev_delta_w = Array.zeros((HIDDEN_SIZE, INPUT_SIZE))
    prev_delta_b = Array.zeros(HIDDEN_SIZE)
    w = Array(layer.W.tolist())
    b = Array(layer.b.tolist())

    for _ in range(5):
        grad_w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
        grad_b_data = _random_vector(rng, HIDDEN_SIZE)

        layer._grad_W = np.array(grad_w_data)
        layer._grad_b = np.array(grad_b_data)
        layer.apply_accumulated_gradient(learning_rate, batch_size=1)

        w, b, prev_delta_w, prev_delta_b = layer_momentum_apply_accumulated_gradient(
            w, b, Array(grad_w_data), Array(grad_b_data), prev_delta_w, prev_delta_b,
            MOMENTUM, learning_rate, 1,
        )

        assert _to_numpy(w) == pytest.approx(layer.W)
        assert _to_numpy(b) == pytest.approx(layer.b)


def test_rejects_batch_size_zero():
    w = Array.zeros((2, 3))
    b = Array.zeros(2)
    with pytest.raises(ValueError):
        layer_momentum_apply_accumulated_gradient(w, b, w, b, w, b, MOMENTUM, 0.1, 0)


def test_rejects_mismatched_shapes():
    w = Array.zeros((2, 3))
    b = Array.zeros(2)
    wrong_shape_prev_delta_w = Array.zeros((3, 2))
    with pytest.raises(ValueError):
        layer_momentum_apply_accumulated_gradient(
            w, b, w, b, wrong_shape_prev_delta_w, b, MOMENTUM, 0.1, 1
        )
