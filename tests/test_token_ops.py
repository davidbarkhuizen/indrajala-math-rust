"""
A patch model's parameter-free token ops (indrajala-ml's layer-norm and attention workplan, D3 and
D8): `patches_*`, a fixed permutation, against numpy's reshape and transpose (indrajala-ml's
`PatchesArrayLayer`), and `token_mean_*` against its left fold over the tokens (`np.cumsum`), by
bits. `Position` has no op: it is `Array`'s `+` and `sum_axis0`, checked here as it uses them.
"""

import numpy as np
import pytest

from indrajala_math_rust import (
    Array,
    patches_downstream,
    patches_forward,
    sum_axis0,
    token_mean_downstream,
    token_mean_forward,
)

# (height, width, channels, patch size)
GRIDS = [(1, 1, 1, 1), (4, 6, 1, 2), (28, 28, 1, 7), (6, 4, 3, 2), (9, 9, 2, 3), (5, 5, 1, 5)]
ROWS = [None, 1, 3]  # None: a single example
# (tokens, features)
TOKENS = [(1, 1), (1, 7), (16, 32), (5, 3), (9, 33)]


def _numpy(array: Array) -> np.ndarray:
    return np.array(array.tolist())


def _bits(values: np.ndarray) -> list[bytes]:
    # by bits, so -0.0 and 0.0 differ, where np.array_equal takes them as equal
    return [v.tobytes() for v in np.ravel(values).astype(np.float64)]


def _batch(values: np.ndarray, rows: int | None) -> np.ndarray:
    return values[0] if rows is None else values


def _patches_reference(x: np.ndarray, height: int, width: int, channels: int, p: int) -> np.ndarray:
    # PatchesArrayLayer.forward_batch: (N, C, U, p, V, p) to (N, U, V, C, p, p)
    n = x.shape[0]
    images = x.reshape(n, channels, height // p, p, width // p, p)
    return images.transpose(0, 2, 4, 1, 3, 5).reshape(n, -1)


def _patches_downstream_reference(delta: np.ndarray, height: int, width: int, channels: int, p: int) -> np.ndarray:
    n = delta.shape[0]
    tokens = delta.reshape(n, height // p, width // p, channels, p, p)
    return tokens.transpose(0, 3, 1, 4, 2, 5).reshape(n, -1)


@pytest.mark.parametrize("height,width,channels,p", GRIDS)
@pytest.mark.parametrize("rows", ROWS)
def test_patches_are_numpys_permutation_and_the_downstream_its_inverse(
    height: int, width: int, channels: int, p: int, rows: int | None
):
    rng = np.random.default_rng(height * 1000 + width * 10 + channels)
    x = rng.uniform(-1.0, 1.0, (rows or 1, height * width * channels))
    out = _numpy(patches_forward(Array(_batch(x, rows).tolist()), height, width, channels, p))
    assert out.shape == _batch(x, rows).shape
    assert _bits(out) == _bits(_patches_reference(x, height, width, channels, p))

    delta = rng.uniform(-1.0, 1.0, x.shape)
    dx = _numpy(patches_downstream(Array(_batch(delta, rows).tolist()), height, width, channels, p))
    assert _bits(dx) == _bits(_patches_downstream_reference(delta, height, width, channels, p))
    # the inverse: the downstream of the forward's output is the input
    back = patches_downstream(
        patches_forward(Array(x.tolist()), height, width, channels, p), height, width, channels, p
    )
    assert _bits(_numpy(back)) == _bits(x)


def test_mnist_patches_put_each_7x7_patch_in_its_token():
    # pixel (h, w) of a 28 x 28 image is feature (h % 7) * 7 + (w % 7) of token (h // 7) * 4 + w // 7
    x = np.arange(28 * 28, dtype=np.float64)
    out = _numpy(patches_forward(Array(x.tolist()), 28, 28, 1, 7)).reshape(16, 49)
    for h, w in [(0, 0), (6, 6), (7, 0), (13, 20), (27, 27)]:
        assert out[(h // 7) * 4 + w // 7, (h % 7) * 7 + w % 7] == h * 28 + w


@pytest.mark.parametrize("tokens,features", TOKENS)
@pytest.mark.parametrize("rows", ROWS)
def test_the_token_mean_is_a_left_fold_over_the_tokens_and_its_downstream_shares_delta(
    tokens: int, features: int, rows: int | None
):
    rng = np.random.default_rng(tokens * 100 + features)
    n = rows or 1
    x = rng.uniform(-1.0, 1.0, (n, tokens * features))
    out = _numpy(token_mean_forward(Array(_batch(x, rows).tolist()), tokens))
    reference = np.cumsum(x.reshape(n, tokens, features), axis=1)[:, -1, :] / tokens
    assert out.shape == _batch(reference, rows).shape
    assert _bits(out) == _bits(reference)

    delta = rng.uniform(-1.0, 1.0, (n, features))
    dx = _numpy(token_mean_downstream(Array(_batch(delta, rows).tolist()), tokens))
    share = (delta / tokens)[:, np.newaxis, :]
    assert dx.shape == ((tokens * features,) if rows is None else (n, tokens * features))
    assert _bits(dx) == _bits(np.broadcast_to(share, (n, tokens, features)))


def test_a_position_table_is_added_with_plus_and_its_gradient_summed_with_sum_axis0():
    rng = np.random.default_rng(7)
    x, table, delta = rng.uniform(-1.0, 1.0, (5, 12)), rng.uniform(-1.0, 1.0, 12), rng.uniform(-1.0, 1.0, (5, 12))
    out = Array(x.tolist()) + Array(table.tolist())
    assert _bits(_numpy(out)) == _bits(x + table)
    assert _bits(_numpy(sum_axis0(Array(delta.tolist())))) == _bits(np.cumsum(delta, axis=0)[-1])


def test_the_token_ops_refuse_shapes_that_dont_fit():
    with pytest.raises(ValueError, match="patches_forward requires a patch_size of 1 or more dividing"):
        patches_forward(Array([0.0] * 36), 6, 6, 1, 4)
    with pytest.raises(ValueError, match="patches_forward requires a patch_size of 1 or more dividing"):
        patches_forward(Array([0.0] * 36), 6, 6, 1, 0)
    with pytest.raises(ValueError, match=r"patches_downstream requires delta of 36 values per example \(6 x 6 x 1\)"):
        patches_downstream(Array([0.0] * 35), 6, 6, 1, 3)
    with pytest.raises(ValueError, match="token_mean_forward requires x of tokens x features values"):
        token_mean_forward(Array([0.0] * 10), 3)
    with pytest.raises(ValueError, match="token_mean_downstream requires 1 or more tokens"):
        token_mean_downstream(Array([0.0] * 3), 0)
