"""
AdamW's decoupled weight decay on layer_adam_apply_accumulated_gradient (indrajala-ml's
training-recipe workplan, D9): W * (1 - learning_rate * weight_decay) - learning_rate * m_hat /
(sqrt(v_hat) + epsilon), PyTorch's AdamW grouping, on W only; at weight_decay 0 Adam by bits.
"""

import math
import random

import pytest

from indrajala_math_rust import Array, layer_adam_apply_accumulated_gradient

BETA1, BETA2, EPSILON, RATE, BATCH = 0.9, 0.999, 1e-8, 0.01, 4


def _arrays(seed: str, rows: int, columns: int) -> tuple[list[list[list[float]]], list[list[float]]]:
    rng = random.Random(seed)
    w, gw, mw = ([[rng.uniform(-1, 1) for _ in range(columns)] for _ in range(rows)] for _ in range(3))
    vw = [[rng.uniform(0, 1) for _ in range(columns)] for _ in range(rows)]
    b, gb, mb = ([rng.uniform(-1, 1) for _ in range(rows)] for _ in range(3))
    vb = [rng.uniform(0, 1) for _ in range(rows)]
    return [w, gw, mw, vw], [b, gb, mb, vb]


def _call(matrices, vectors, t, **weight_decay):
    w, gw, mw, vw = (Array(x) for x in matrices)
    b, gb, mb, vb = (Array(x) for x in vectors)
    result = layer_adam_apply_accumulated_gradient(
        w, b, gw, gb, mw, vw, mb, vb, t, BETA1, BETA2, EPSILON, RATE, BATCH, **weight_decay
    )
    return [array.tolist() for array in result]


def _adamw(param, grad, m, v, t, weight_decay):
    c1, c2 = 1.0 - BETA1**t, 1.0 - BETA2**t
    decay = 1.0 - RATE * weight_decay
    out = []
    for p, g, mi, vi in zip(param, grad, m, v):
        g = g / BATCH
        new_m = BETA1 * mi + (1.0 - BETA1) * g
        new_v = BETA2 * vi + (1.0 - BETA2) * g * g
        out.append(p * decay - RATE * (new_m / c1) / (math.sqrt(new_v / c2) + EPSILON))
    return out


@pytest.mark.parametrize("t", [1, 2, 3])
@pytest.mark.parametrize("weight_decay", [0.1, 0.01, 1.0])
def test_weight_decay_shrinks_w_before_the_adam_step_and_leaves_b_by_bits(t: int, weight_decay: float):
    # t of 1 to 3, where ** and the crate's square-and-multiply agree (tests/test_adam_powi.py)
    matrices, vectors = _arrays(f"adamw {t} {weight_decay}", 3, 5)
    new_w, new_b, *_ = _call(matrices, vectors, t, weight_decay=weight_decay)
    w, gw, mw, vw = matrices
    for row, w_row, gw_row, mw_row, vw_row in zip(new_w, w, gw, mw, vw):
        assert row == _adamw(w_row, gw_row, mw_row, vw_row, t, weight_decay)
    b, gb, mb, vb = vectors
    assert new_b == _adamw(b, gb, mb, vb, t, 0.0)
    assert new_b == _call(matrices, vectors, t)[1]


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (64, 64), (65, 257)])
def test_weight_decay_zero_is_adam_by_bits(shape: tuple[int, int]):
    matrices, vectors = _arrays(f"adam {shape}", *shape)
    assert _call(matrices, vectors, 7, weight_decay=0.0) == _call(matrices, vectors, 7)


def test_weight_decay_moves_w_toward_zero_where_adam_would_not():
    matrices, vectors = _arrays("decay", 2, 3)
    adam, adamw = _call(matrices, vectors, 1), _call(matrices, vectors, 1, weight_decay=0.5)
    shrink = [[a - RATE * 0.5 * w for a, w in zip(arow, wrow)] for arow, wrow in zip(adam[0], matrices[0])]
    for row, expected in zip(adamw[0], shrink):
        assert row == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("weight_decay", [-0.1, math.inf, math.nan])
def test_a_weight_decay_that_cannot_train_is_refused(weight_decay: float):
    matrices, vectors = _arrays("refused", 2, 2)
    with pytest.raises(ValueError, match="weight_decay"):
        _call(matrices, vectors, 1, weight_decay=weight_decay)
