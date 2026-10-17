"""
Adam's bias corrections raise beta1 and beta2 to the step t by square-and-multiply from t's low
bit, compiler-rt's __powidf2 order, written out in src/scalar.rs (indrajala-ml's aarch64
workplan, D6) rather than left to how f64::powi lowers. Checked by bits against fused.rs's update
replicated here, at every t to 64 and at large t.
"""

import math
import random

import pytest

from indrajala_math_rust import Array, layer_adam_apply_accumulated_gradient


def _powi(base: float, t: int) -> float:
    r = 1.0
    while True:
        if t & 1:
            r *= base
        t >>= 1
        if t == 0:
            return r
        base *= base


def _adam(param, grad, m, v, t, beta1, beta2, epsilon, learning_rate, batch_size):
    c1, c2 = 1.0 - _powi(beta1, t), 1.0 - _powi(beta2, t)
    out = ([], [], [])
    for p, g, mi, vi in zip(param, grad, m, v):
        g = g / batch_size
        new_m = beta1 * mi + (1.0 - beta1) * g
        new_v = beta2 * vi + (1.0 - beta2) * g * g
        out[0].append(p - learning_rate * (new_m / c1) / (math.sqrt(new_v / c2) + epsilon))
        out[1].append(new_m)
        out[2].append(new_v)
    return out


@pytest.mark.parametrize("t", [*range(1, 65), 100, 1_000, 10_000, 100_000, 1_000_000])
def test_adam_bias_corrections_are_square_and_multiply_by_bits(t: int):
    rng = random.Random(f"adam {t}")
    w = [[rng.uniform(-1, 1) for _ in range(4)] for _ in range(3)]
    gw, mw = ([[rng.uniform(-1, 1) for _ in range(4)] for _ in range(3)] for _ in range(2))
    vw = [[rng.uniform(0, 1) for _ in range(4)] for _ in range(3)]
    b, gb, mb = ([rng.uniform(-1, 1) for _ in range(3)] for _ in range(3))
    vb = [rng.uniform(0, 1) for _ in range(3)]
    new_w, new_b, new_mw, new_vw, new_mb, new_vb = layer_adam_apply_accumulated_gradient(
        Array(w), Array(b), Array(gw), Array(gb), Array(mw), Array(vw), Array(mb), Array(vb),
        t, 0.9, 0.999, 1e-8, 0.001, 2,
    )  # fmt: skip
    flat = lambda rows: [x for row in rows for x in row]
    expected_w = _adam(flat(w), flat(gw), flat(mw), flat(vw), t, 0.9, 0.999, 1e-8, 0.001, 2.0)
    expected_b = _adam(b, gb, mb, vb, t, 0.9, 0.999, 1e-8, 0.001, 2.0)
    assert (flat(new_w.tolist()), flat(new_mw.tolist()), flat(new_vw.tolist())) == expected_w
    assert (new_b.tolist(), new_mb.tolist(), new_vb.tolist()) == expected_b
