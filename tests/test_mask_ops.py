"""
The sigmoid and dropout mask ops (indrajala-ml's layer-norm and attention workplan, D5): a hidden
layer's delta from a downstream it was handed, for the layer right before a `LayerNorm`, which has
no `W` for the fused `layer_*hidden_delta*` ops to read. Each is checked by bits against the fused
op given the downstream that op computes (`layer_downstream*`), and against numpy's elementwise
expression in the same grouping, which is correctly rounded, so the bits must agree.
"""

import numpy as np
import pytest

from indrajala_math_rust import (
    Array,
    array_dropout_mask,
    array_sigmoid_mask,
    layer_downstream,
    layer_downstream_batch,
    layer_dropout_hidden_delta,
    layer_dropout_hidden_delta_batch,
    layer_hidden_delta,
    layer_hidden_delta_batch,
)

# (next size, size): the next layer's and this layer's
SIZES = [(1, 1), (3, 2), (8, 5), (33, 7)]
ROWS = [None, 1, 2, 9]  # None: a single example
KEEP = 0.75


def _numpy(array: Array) -> np.ndarray:
    return np.array(array.tolist())


def _bits(values: np.ndarray) -> list[bytes]:
    # by bits, so -0.0 and 0.0 differ, where np.array_equal takes them as equal
    return [v.tobytes() for v in np.ravel(values).astype(np.float64)]


def _arrays(seed: int, next_size: int, size: int, rows: int | None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (size,) if rows is None else (rows, size)
    next_shape = (next_size,) if rows is None else (rows, next_size)
    return {
        "next_w": rng.uniform(-1.0, 1.0, (next_size, size)),
        "next_delta": rng.uniform(-1.0, 1.0, next_shape),
        "a": rng.uniform(0.0, 1.0, shape),
        "mask": (rng.uniform(0.0, 1.0, shape) < KEEP).astype(np.float64),
    }


def _downstream(c: dict[str, np.ndarray], rows: int | None) -> Array:
    w, delta = Array(c["next_w"].tolist()), Array(c["next_delta"].tolist())
    return layer_downstream(w, delta) if rows is None else layer_downstream_batch(w, delta)


@pytest.mark.parametrize("next_size,size", SIZES)
@pytest.mark.parametrize("rows", ROWS)
def test_the_sigmoid_mask_of_the_downstream_is_the_fused_hidden_delta(next_size: int, size: int, rows: int | None):
    c = _arrays(next_size * 100 + size, next_size, size, rows)
    downstream = _downstream(c, rows)
    a = Array(c["a"].tolist())
    w, delta = Array(c["next_w"].tolist()), Array(c["next_delta"].tolist())
    fused = layer_hidden_delta(w, delta, a) if rows is None else layer_hidden_delta_batch(w, delta, a)
    masked = _numpy(array_sigmoid_mask(downstream, a))
    assert _bits(masked) == _bits(_numpy(fused))
    assert _bits(masked) == _bits(_numpy(downstream) * c["a"] * (1.0 - c["a"]))


@pytest.mark.parametrize("next_size,size", SIZES)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("was_training", [True, False])
def test_the_dropout_mask_of_the_downstream_is_the_fused_hidden_delta(
    next_size: int, size: int, rows: int | None, was_training: bool
):
    c = _arrays(next_size * 100 + size + 1, next_size, size, rows)
    downstream = _downstream(c, rows)
    base, mask = Array(c["a"].tolist()), Array(c["mask"].tolist())
    w, delta = Array(c["next_w"].tolist()), Array(c["next_delta"].tolist())
    op = layer_dropout_hidden_delta if rows is None else layer_dropout_hidden_delta_batch
    fused = op(w, delta, base, mask, KEEP, was_training)
    masked = _numpy(array_dropout_mask(downstream, base, mask, KEEP, was_training))
    scale = c["mask"] / KEEP if was_training else 1.0
    assert _bits(masked) == _bits(_numpy(fused))
    assert _bits(masked) == _bits(_numpy(downstream) * (c["a"] * (1.0 - c["a"])) * scale)


def test_the_mask_ops_refuse_arrays_of_other_shapes():
    two, three = Array([0.5, 0.5]), Array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="array_sigmoid_mask requires matching shapes"):
        array_sigmoid_mask(two, three)
    with pytest.raises(ValueError, match="array_dropout_mask requires matching shapes"):
        array_dropout_mask(two, two, three, KEEP, True)
    with pytest.raises(ValueError, match="array_dropout_mask requires matching shapes"):
        array_dropout_mask(three, two, two, KEEP, True)
