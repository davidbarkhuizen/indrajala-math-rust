"""
The residual-block ops (indrajala-ml's residual-connections workplan, D4 and D8): the affine layer's
forward pass (`affine_forward*`), and the fused hidden deltas of a layer before a block's fork
(`layer_*hidden_delta_skip*`), `(body_delta @ body_W + skip) * f'(a)`. Each skip op is checked by
bits against the crate's own unfused downstream (`layer_downstream*`), then the add and the
activation's derivative in numpy, which are correctly rounded elementwise operations in the ops'
grouping, so the bits must agree.
"""

import numpy as np
import pytest

from indrajala_math_rust import (
    Array,
    affine_forward,
    affine_forward_batch,
    exp,
    layer_downstream,
    layer_downstream_batch,
    layer_dropout_hidden_delta,
    layer_dropout_hidden_delta_batch,
    layer_dropout_hidden_delta_skip,
    layer_dropout_hidden_delta_skip_batch,
    layer_forward,
    layer_forward_batch,
    layer_hidden_delta,
    layer_hidden_delta_batch,
    layer_hidden_delta_skip,
    layer_hidden_delta_skip_batch,
    layer_relu_hidden_delta,
    layer_relu_hidden_delta_batch,
    layer_relu_hidden_delta_skip,
    layer_relu_hidden_delta_skip_batch,
)

# (body size, size): the body's first layer's size and the block's
SIZES = [(1, 1), (3, 2), (8, 5), (33, 7)]
BATCHES = [1, 2, 9]


def _numpy(array: Array) -> np.ndarray:
    return np.array(array.tolist())


def _bits(values: np.ndarray) -> list[bytes]:
    # by bits, so -0.0 and 0.0 differ, where np.array_equal takes them as equal
    return [v.tobytes() for v in np.ravel(values).astype(np.float64)]


def _arrays(seed: int, body_size: int, size: int, rows: int | None) -> dict[str, np.ndarray]:
    # the body's first layer's W and delta, the add's delta (skip), and this layer's activations,
    # single-example (rows None) or a batch
    rng = np.random.default_rng(seed)
    shape = (size,) if rows is None else (rows, size)
    body_shape = (body_size,) if rows is None else (rows, body_size)
    a = rng.uniform(0.0, 1.0, shape)
    a[..., 0] = 0.0  # ReLU's boundary: inactive
    return {
        "body_w": rng.uniform(-1.0, 1.0, (body_size, size)),
        "body_delta": rng.uniform(-1.0, 1.0, body_shape),
        "skip": rng.uniform(-1.0, 1.0, shape),
        "a": a,
        "mask": (rng.uniform(0.0, 1.0, shape) >= 0.3).astype(np.float64),
    }


def _array(values: np.ndarray) -> Array:
    return Array(values.tolist())


def _downstream(arrays: dict[str, np.ndarray], batch: bool) -> np.ndarray:
    w, delta = _array(arrays["body_w"]), _array(arrays["body_delta"])
    return _numpy(layer_downstream_batch(w, delta) if batch else layer_downstream(w, delta))


@pytest.mark.parametrize("rows", [None, *BATCHES])
@pytest.mark.parametrize("body_size, size", SIZES)
def test_the_sigmoid_skip_delta_is_the_downstream_plus_skip_times_the_derivative(
    body_size: int, size: int, rows: int | None
):
    arrays = _arrays(1, body_size, size, rows)
    ds = _downstream(arrays, rows is not None) + arrays["skip"]
    a = arrays["a"]
    op = layer_hidden_delta_skip if rows is None else layer_hidden_delta_skip_batch
    actual = op(_array(arrays["body_w"]), _array(arrays["body_delta"]), _array(arrays["skip"]), _array(a))

    assert _bits(_numpy(actual)) == _bits(ds * a * (1.0 - a))


@pytest.mark.parametrize("rows", [None, *BATCHES])
@pytest.mark.parametrize("body_size, size", SIZES)
def test_the_relu_skip_delta_masks_the_downstream_plus_skip(body_size: int, size: int, rows: int | None):
    arrays = _arrays(2, body_size, size, rows)
    ds = _downstream(arrays, rows is not None) + arrays["skip"]
    a = arrays["a"]
    op = layer_relu_hidden_delta_skip if rows is None else layer_relu_hidden_delta_skip_batch
    actual = op(_array(arrays["body_w"]), _array(arrays["body_delta"]), _array(arrays["skip"]), _array(a))

    # an inactive unit's delta is +0.0, as layer_relu_hidden_delta's
    assert _bits(_numpy(actual)) == _bits(np.where(a > 0.0, ds, 0.0))


@pytest.mark.parametrize("was_training", [True, False])
@pytest.mark.parametrize("rows", [None, *BATCHES])
@pytest.mark.parametrize("body_size, size", SIZES)
def test_the_dropout_skip_delta_scales_the_downstream_plus_skip(
    body_size: int, size: int, rows: int | None, was_training: bool
):
    arrays = _arrays(3, body_size, size, rows)
    ds = _downstream(arrays, rows is not None) + arrays["skip"]
    base, mask, keep = arrays["a"], arrays["mask"], 0.7
    op = layer_dropout_hidden_delta_skip if rows is None else layer_dropout_hidden_delta_skip_batch
    actual = op(
        _array(arrays["body_w"]),
        _array(arrays["body_delta"]),
        _array(arrays["skip"]),
        _array(base),
        _array(mask),
        keep,
        was_training,
    )

    scale = mask / keep if was_training else 1.0
    assert _bits(_numpy(actual)) == _bits(ds * (base * (1.0 - base)) * scale)


@pytest.mark.parametrize("rows", [None, 3])
def test_a_zero_skip_gives_the_ops_without_it(rows: int | None):
    # the skip is one addition before the derivative, so with skip 0 each op is its non-skip form
    # (up to the sign of a zero: -0.0 + 0.0 is 0.0)
    arrays = _arrays(4, 6, 4, rows)
    w, delta, a, mask = (_array(arrays[name]) for name in ("body_w", "body_delta", "a", "mask"))
    zero = _array(np.zeros_like(arrays["skip"]))
    batch = rows is not None
    pairs = [
        (
            (layer_hidden_delta_skip_batch if batch else layer_hidden_delta_skip)(w, delta, zero, a),
            (layer_hidden_delta_batch if batch else layer_hidden_delta)(w, delta, a),
        ),
        (
            (layer_relu_hidden_delta_skip_batch if batch else layer_relu_hidden_delta_skip)(w, delta, zero, a),
            (layer_relu_hidden_delta_batch if batch else layer_relu_hidden_delta)(w, delta, a),
        ),
        (
            (layer_dropout_hidden_delta_skip_batch if batch else layer_dropout_hidden_delta_skip)(
                w, delta, zero, a, mask, 0.7, True
            ),
            (layer_dropout_hidden_delta_batch if batch else layer_dropout_hidden_delta)(w, delta, a, mask, 0.7, True),
        ),
    ]
    for skip, plain in pairs:
        assert np.array_equal(_numpy(skip), _numpy(plain))


def test_the_skip_ops_refuse_a_skip_of_another_shape():
    w, delta, a = Array.zeros((3, 2)), Array.zeros(3), Array.zeros(2)
    with pytest.raises(ValueError, match="matching shapes"):
        layer_hidden_delta_skip(w, delta, Array.zeros(3), a)
    with pytest.raises(ValueError, match="matching shapes"):
        layer_relu_hidden_delta_skip_batch(w, Array.zeros((4, 3)), Array.zeros((4, 3)), Array.zeros((4, 2)))
    with pytest.raises(ValueError, match="matching shapes"):
        layer_dropout_hidden_delta_skip(w, delta, Array.zeros(1), a, a, 0.5, True)


@pytest.mark.parametrize("size, input_size", [(1, 1), (4, 3), (7, 33)])
def test_the_affine_forward_pass_is_the_dense_layers_pre_activation(size: int, input_size: int):
    rng = np.random.default_rng(size)
    w = _array(rng.uniform(-1.0, 1.0, (size, input_size)))
    b = _array(rng.uniform(-1.0, 1.0, size))
    x_rows = rng.uniform(-1.0, 1.0, (5, input_size))

    batch = _numpy(affine_forward_batch(w, _array(x_rows), b))
    sigmoid_batch = _numpy(layer_forward_batch(w, _array(x_rows), b))
    for i, x_row in enumerate(x_rows):
        z = affine_forward(w, _array(x_row), b)
        assert _bits(_numpy(z)) == _bits(_numpy(w @ _array(x_row) + b))
        # row i of the batch is the single-example product, as for layer_forward_batch
        assert _bits(batch[i]) == _bits(_numpy(z))
        # and the sigmoid layer's activation is the sigmoid of it, with the crate's exp
        sigmoid = 1.0 / (1.0 + _numpy(exp(_array(-_numpy(z)))))
        assert _bits(_numpy(layer_forward(w, _array(x_row), b))) == _bits(sigmoid)
        assert _bits(sigmoid_batch[i]) == _bits(sigmoid)
