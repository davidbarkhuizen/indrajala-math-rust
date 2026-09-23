"""
`layer_adam_apply_accumulated_gradient` is one fused Rust call for the whole Adam (Kingma & Ba,
2014) update rule per parameter, checked
against `indrajala_ml.model.adam_array_layer.AdamArrayLayer` - the actual production reference
this function replaces - the same treatment `test_fused_layer_ops.py` gives every non-Adam fused
op.
"""

import random

import numpy as np
import pytest

from indrajala_ml_array import Array, layer_adam_apply_accumulated_gradient

from indrajala_ml.model.adam_array_layer import AdamArrayLayer

SEEDS = range(30)
INPUT_SIZE = 8
HIDDEN_SIZE = 5
BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-8


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
def test_layer_adam_apply_accumulated_gradient_matches_adam_array_layer_at_step_one(seed, batch_size):
    rng = random.Random(seed)
    w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    b_data = _random_vector(rng, HIDDEN_SIZE)
    grad_w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
    grad_b_data = _random_vector(rng, HIDDEN_SIZE)
    learning_rate = rng.uniform(0.001, 1.0)

    layer = AdamArrayLayer(HIDDEN_SIZE, INPUT_SIZE, BETA1, BETA2, EPSILON)
    layer.W, layer.b = np.array(w_data), np.array(b_data)
    layer._grad_W, layer._grad_b = np.array(grad_w_data), np.array(grad_b_data)
    layer.apply_accumulated_gradient(learning_rate, batch_size)

    new_w, new_b, new_m_w, new_v_w, new_m_b, new_v_b = layer_adam_apply_accumulated_gradient(
        Array(w_data),
        Array(b_data),
        Array(grad_w_data),
        Array(grad_b_data),
        Array.zeros((HIDDEN_SIZE, INPUT_SIZE)),
        Array.zeros((HIDDEN_SIZE, INPUT_SIZE)),
        Array.zeros(HIDDEN_SIZE),
        Array.zeros(HIDDEN_SIZE),
        1,
        BETA1,
        BETA2,
        EPSILON,
        learning_rate,
        batch_size,
    )
    assert _to_numpy(new_w) == pytest.approx(layer.W)
    assert _to_numpy(new_b) == pytest.approx(layer.b)
    assert _to_numpy(new_m_w) == pytest.approx(layer._m_W)
    assert _to_numpy(new_v_w) == pytest.approx(layer._v_W)
    assert _to_numpy(new_m_b) == pytest.approx(layer._m_b)
    assert _to_numpy(new_v_b) == pytest.approx(layer._v_b)


@pytest.mark.parametrize("seed", SEEDS)
def test_layer_adam_apply_accumulated_gradient_matches_adam_array_layer_across_several_steps(seed):
    # m/v/t only actually exercise their accumulation logic across repeated steps, unlike a
    # stateless update where a single comparison at t=1 would do - mirrors
    # test_adam_array_layer.py's own reasoning for using a multi-step sweep, not just one call.
    rng = random.Random(seed)
    layer = AdamArrayLayer(HIDDEN_SIZE, INPUT_SIZE, BETA1, BETA2, EPSILON)
    layer.W = np.array(_random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE))
    layer.b = np.array(_random_vector(rng, HIDDEN_SIZE))
    learning_rate = rng.uniform(0.001, 1.0)

    m_w = Array.zeros((HIDDEN_SIZE, INPUT_SIZE))
    v_w = Array.zeros((HIDDEN_SIZE, INPUT_SIZE))
    m_b = Array.zeros(HIDDEN_SIZE)
    v_b = Array.zeros(HIDDEN_SIZE)
    w = Array(layer.W.tolist())
    b = Array(layer.b.tolist())

    for t in range(1, 6):
        grad_w_data = _random_matrix(rng, HIDDEN_SIZE, INPUT_SIZE)
        grad_b_data = _random_vector(rng, HIDDEN_SIZE)

        layer._grad_W = np.array(grad_w_data)
        layer._grad_b = np.array(grad_b_data)
        layer.apply_accumulated_gradient(learning_rate, batch_size=1)

        w, b, m_w, v_w, m_b, v_b = layer_adam_apply_accumulated_gradient(
            w, b, Array(grad_w_data), Array(grad_b_data), m_w, v_w, m_b, v_b,
            t, BETA1, BETA2, EPSILON, learning_rate, 1,
        )

        assert _to_numpy(w) == pytest.approx(layer.W)
        assert _to_numpy(b) == pytest.approx(layer.b)
        assert layer._t == t


def test_rejects_batch_size_zero():
    w = Array.zeros((2, 3))
    b = Array.zeros(2)
    with pytest.raises(ValueError):
        layer_adam_apply_accumulated_gradient(
            w, b, w, b, w, w, b, b, 1, BETA1, BETA2, EPSILON, 0.1, 0
        )


def test_rejects_t_zero():
    w = Array.zeros((2, 3))
    b = Array.zeros(2)
    with pytest.raises(ValueError):
        layer_adam_apply_accumulated_gradient(
            w, b, w, b, w, w, b, b, 0, BETA1, BETA2, EPSILON, 0.1, 1
        )


def test_rejects_mismatched_shapes():
    w = Array.zeros((2, 3))
    b = Array.zeros(2)
    wrong_shape_m_w = Array.zeros((3, 2))
    with pytest.raises(ValueError):
        layer_adam_apply_accumulated_gradient(
            w, b, w, b, wrong_shape_m_w, w, b, b, 1, BETA1, BETA2, EPSILON, 0.1, 1
        )
