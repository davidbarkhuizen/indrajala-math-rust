"""
The dense batched forward ops (`layer_*forward_batch`), which compute `X @ W.T` with `matmul_nt`:
without copying `W.T`, as one `dot_product(W[j], X[i])` per output. That makes every row of a
batched forward bit-identical to the single-example forward on that row, which these tests pin
exactly, on both sides of the threading threshold (8M flops).
"""

import numpy as np
import pytest

from indrajala_math_rust import (
    Array,
    layer_dropout_forward,
    layer_dropout_forward_batch,
    layer_forward,
    layer_forward_batch,
    layer_relu_forward,
    layer_relu_forward_batch,
    layer_softmax_forward,
    layer_softmax_forward_batch,
    matmul_threads_for,
    set_matmul_threading,
)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _dropout_eval_forward(w, x, b):
    return layer_dropout_forward(w, x, b, 0.5, False)[0]


def _dropout_eval_forward_batch(w, x, b):
    return layer_dropout_forward_batch(w, x, b, 0.5, False)[0]


# (name, single-example op, batched op, numpy reference for the batch)
VARIANTS = [
    ("sigmoid", layer_forward, layer_forward_batch, _sigmoid),
    ("relu", layer_relu_forward, layer_relu_forward_batch, lambda z: np.maximum(z, 0.0)),
    ("softmax", layer_softmax_forward, layer_softmax_forward_batch, _softmax),
    ("dropout eval", _dropout_eval_forward, _dropout_eval_forward_batch, _sigmoid),
]

# (batch, input_size, size). The last two are over the threading threshold (batch * input_size *
# size >= 8M flops), so their rows are split across threads
SHAPES = [
    (1, 1, 1),
    (1, 7, 1),
    (3, 1, 5),
    (1, 30, 10),
    (5, 13, 9),
    (32, 784, 30),
    (7, 5408, 32),
    (512, 30, 10),
    (256, 784, 30),
    (32, 5408, 32),
    (3, 5408, 300),
    (64, 5408, 32),
    (512, 784, 30),
]
THREADED_SHAPES = SHAPES[-2:]


def _inputs(batch, input_size, size):
    rng = np.random.default_rng(batch * 1_000_000 + input_size * 1000 + size)
    W = rng.uniform(-0.3, 0.3, size=(size, input_size))
    X = rng.uniform(-1.0, 1.0, size=(batch, input_size))
    b = rng.uniform(-0.3, 0.3, size=size)
    return W, X, b


@pytest.mark.parametrize("name, forward, forward_batch, reference", VARIANTS)
@pytest.mark.parametrize("batch, input_size, size", SHAPES)
def test_forward_batch_matches_numpy(name, forward, forward_batch, reference, batch, input_size, size):
    W, X, b = _inputs(batch, input_size, size)
    result = forward_batch(Array(W.tolist()), Array(X.tolist()), Array(b.tolist()))
    assert result.shape == (batch, size)
    # numpy sums in another order, so an output that cancels to near zero differs by a few ULPs of
    # its terms' size, not its own: the absolute tolerance scales with sum(|x * w|) + |b|
    terms = np.abs(X) @ np.abs(W).T + np.abs(b)
    expected = reference(X @ W.T + b)
    assert np.all(np.abs(np.array(result.tolist()) - expected) <= 1e-12 * np.abs(expected) + 1e-15 * terms)


@pytest.mark.parametrize("name, forward, forward_batch, reference", VARIANTS)
@pytest.mark.parametrize("batch, input_size, size", SHAPES)
def test_forward_batch_rows_are_bit_identical_to_forward(
    name, forward, forward_batch, reference, batch, input_size, size
):
    W, X, b = _inputs(batch, input_size, size)
    w, b_array = Array(W.tolist()), Array(b.tolist())
    rows = forward_batch(w, Array(X.tolist()), b_array).tolist()
    for i in range(batch):
        assert rows[i] == forward(w, Array(X[i].tolist()), b_array).tolist()


def test_forward_batch_rejects_mismatched_shapes():
    W = Array.zeros((4, 7))
    b = Array.zeros(4)
    with pytest.raises(ValueError):
        layer_forward_batch(W, Array.zeros((3, 6)), b)
    with pytest.raises(ValueError):
        layer_forward_batch(W, Array.zeros(7), b)
    with pytest.raises(ValueError):
        layer_forward_batch(W, Array.zeros((3, 7)), Array.zeros(5))


def test_the_shapes_cover_both_sides_of_the_threading_threshold():
    # at the default threshold, with 8 threads allowed on any machine, THREADED_SHAPES split
    # across threads and the rest don't, so a threshold change can't silently drop the threaded path
    set_matmul_threading(8, 0)
    try:
        for batch, input_size, size in SHAPES:
            expected = 8 if (batch, input_size, size) in THREADED_SHAPES else 1
            assert matmul_threads_for(batch, input_size, size) == expected, (batch, input_size, size)
    finally:
        set_matmul_threading(0, 0)
