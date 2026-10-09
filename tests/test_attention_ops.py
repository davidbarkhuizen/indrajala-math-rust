"""
The attention ops (`attention_*`, indrajala-ml's multi-head attention workplan, D4): one call per
pass, each composed of the blocks project, attend and combine. Each block's outputs are checked by
bits against the crate's own unfused composition of indrajala-ml's README expressions (Layer norm
and attention), built here from the crate's existing ops: the projections, the downstream products
and the gradients as a dense layer's (`affine_forward_batch`, `layer_downstream_batch`,
`layer_accumulate_gradient_batch`) on the `(N * T, d)` and `(N * T, h * d_k)` rows; the products
between activations as `@` per example and head; the row softmax as `array_softmax`; the softmax
backward's row sums as `sum_axis0` of the transpose, a left fold. The elementwise steps between
them, and the heads' packing side by side, are numpy's, exact. numpy's own products (BLAS) are not
the crate's, so numpy is the reference only where no product is involved, and in the exact tests.

At one head the ops are also pinned to the single-head ops' outputs, recorded before the multi-head
change, which the unmasked ops still compute after the causal mask (indrajala-ml's sequence task
workplan, D7). The composition masks its scores with numpy, exact, before `array_softmax`.
"""

import hashlib
import math

import numpy as np
import pytest

from indrajala_math_rust import (
    Array,
    affine_forward_batch,
    array_softmax,
    attention_accumulate_gradient_batch,
    attention_downstream_batch,
    attention_forward,
    attention_forward_batch,
    layer_accumulate_gradient_batch,
    layer_downstream_batch,
    sum_axis0,
    token_mean_forward,
)

# (examples, tokens, features, heads, key_size)
ONE_HEAD = [
    # one token, the patch model's 16 x 32, and widths off the 4 lanes
    (1, 1, 1, 1, 1),
    (2, 1, 3, 1, 3),
    (1, 4, 3, 1, 3),
    (3, 5, 6, 1, 6),
    (2, 16, 32, 1, 32),
    (2, 7, 33, 1, 33),
]
MULTI_HEAD = [
    (1, 4, 6, 2, 3),
    (3, 5, 6, 3, 2),
    (2, 16, 32, 4, 8),
    (2, 16, 32, 2, 16),
    # h * d_k above and below d
    (2, 7, 12, 4, 5),
    (2, 5, 6, 2, 2),
    (1, 3, 8, 4, 8),
]
SHAPES = ONE_HEAD + MULTI_HEAD
NAMES = ["wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo"]

# Each one-head case's digests (_digests) from the single-head ops before the multi-head change
# (indrajala-math-rust 66ecc0d): the batch's forward, downstream and gradients, then one example's.
RECORDED = {
    (1, 1, 1): ("39e6ebfd23839f9b", "5557f39c7e445ff3", "124d89b2c0b0316e", "ffea7c77b82a0fb4", "a6123470652114f4", "124d89b2c0b0316e"),
    (2, 1, 3): ("646c3a87283ffc52", "af94255c55c7b49b", "d0f7313f6cc80725", "68ddbc9ec885735f", "43c0147417e461ce", "61a2a9391f40cd7a"),
    (1, 4, 3): ("d48854fef53380cb", "60e0ac518f97871b", "8382597018198860", "e5e20468289c66d1", "5d9b387357a72487", "8382597018198860"),
    (3, 5, 6): ("656737ab2b4ee39a", "81f07c6faf978642", "7afcc55f43dc674d", "e6b857bc875c3029", "3538696d49f1072c", "7c8c865c25397e22"),
    (2, 16, 32): ("895c6107926e20ea", "3fe5146d28d1adcd", "e95f8abcf0e65bdd", "14d0daa2af2b02d3", "da8bab0fc617258a", "2bb227f90d965076"),
    (2, 7, 33): ("83b0126ce1e46d82", "4fc7d6e6602a11b3", "d47489defd89315a", "e9b6bf7fd7e7354d", "313b81336ee547af", "19e42f67776f7843"),
}  # fmt: skip


def _numpy(array: Array) -> np.ndarray:
    return np.array(array.tolist())


def _array(values: np.ndarray) -> Array:
    return Array(values.tolist())


def _bits(values: np.ndarray) -> list[bytes]:
    # by bits, so -0.0 and 0.0 differ, where np.array_equal takes them as equal
    return [v.tobytes() for v in np.ravel(values).astype(np.float64)]


def _id(shape: tuple[int, ...]) -> str:
    return "x".join(map(str, shape))


def _case(examples: int, tokens: int, features: int, heads: int, key_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng([examples, tokens, features, heads, key_size])
    d, width = features, heads * key_size
    c = {"x": rng.uniform(-2.0, 2.0, (examples, tokens * d))}
    for name in NAMES:
        rows, fan_in = (d, width) if name[1] == "o" else (width, d)
        limit = 1.0 / math.sqrt(fan_in)
        c[name] = rng.uniform(-limit, limit, (rows, fan_in) if name[0] == "w" else rows)
    c["delta"] = rng.uniform(-1.0, 1.0, (examples, tokens * d))
    for name in NAMES:
        c["grad_" + name] = rng.uniform(-1.0, 1.0, c[name].shape)
    return c


def _parameters(c: dict[str, np.ndarray]) -> list[Array]:
    return [_array(c[name]) for name in NAMES]


def _head(rows: np.ndarray, n: int, i: int, tokens: int, width: int) -> np.ndarray:
    # example n's head i: rows n * T.., columns i * width..
    return rows[n * tokens : (n + 1) * tokens, i * width : (i + 1) * width]


def _side_by_side(blocks: list[list[np.ndarray]]) -> np.ndarray:
    # blocks[n][i], each (T, width), as (N * T, h * width) rows
    return np.concatenate([np.concatenate(heads, axis=1) for heads in blocks])


def _masked(scores: Array, tokens: int) -> Array:
    # each S_tj with j > t set to -inf
    future = np.triu(np.ones((tokens, tokens), dtype=np.bool_), k=1)
    return _array(np.where(future, -np.inf, _numpy(scores)))


def _composed_forward(c: dict[str, np.ndarray], tokens: int, heads: int, causal: bool = False) -> dict[str, np.ndarray]:
    d, width = c["bo"].shape[0], c["bq"].shape[0]
    d_k = width // heads
    rows = _array(c["x"].reshape(-1, d))
    q, k, v = (_numpy(affine_forward_batch(_array(c["w" + n]), rows, _array(c["b" + n]))) for n in "qkv")
    p: list[list[np.ndarray]] = []
    h: list[list[np.ndarray]] = []
    for n in range(c["x"].shape[0]):
        p.append([])
        h.append([])
        for i in range(heads):
            q_ni, k_ni, v_ni = (_array(_head(m, n, i, tokens, d_k)) for m in (q, k, v))
            scores = (q_ni @ k_ni.T) / math.sqrt(d_k)
            p_ni = array_softmax(_masked(scores, tokens) if causal else scores)
            p[n].append(_numpy(p_ni))
            h[n].append(_numpy(p_ni @ v_ni))
    h_rows = _side_by_side(h)
    a = _numpy(affine_forward_batch(_array(c["wo"]), _array(h_rows), _array(c["bo"])))
    return {"a": a.reshape(c["x"].shape), "q": q, "k": k, "v": v, "p": _side_by_side(p), "h": h_rows}


def _forward(c: dict[str, np.ndarray], heads: int, causal: bool = False) -> dict[str, np.ndarray]:
    result = attention_forward_batch(_array(c["x"]), *_parameters(c), heads=heads, causal=causal)
    return {name: _numpy(array) for name, array in zip(["a", "q", "k", "v", "p", "h"], result)}


def _composed_backward(
    c: dict[str, np.ndarray], f: dict[str, np.ndarray], tokens: int, heads: int
) -> dict[str, np.ndarray]:
    d, width = c["bo"].shape[0], c["bq"].shape[0]
    d_k = width // heads
    s = math.sqrt(d_k)
    dh = _numpy(layer_downstream_batch(_array(c["wo"]), _array(c["delta"].reshape(-1, d))))
    dq: list[list[np.ndarray]] = []
    dk: list[list[np.ndarray]] = []
    dv: list[list[np.ndarray]] = []
    for n in range(c["x"].shape[0]):
        for blocks in (dq, dk, dv):
            blocks.append([])
        for i in range(heads):
            q_ni, k_ni, v_ni, dh_ni = (_array(_head(m, n, i, tokens, d_k)) for m in (f["q"], f["k"], f["v"], dh))
            p = _head(f["p"], n, i, tokens, tokens)
            dp = _numpy(dh_ni @ v_ni.T)
            dv[n].append(_numpy(_array(p).T @ dh_ni))
            r = _numpy(sum_axis0(_array((dp * p).T)))  # each row's left fold
            ds = _array(p * (dp - r[:, np.newaxis]))
            dq[n].append(_numpy((ds @ k_ni) / s))
            dk[n].append(_numpy((ds.T @ q_ni) / s))
    dq_rows, dk_rows, dv_rows = _side_by_side(dq), _side_by_side(dk), _side_by_side(dv)
    dx = (
        layer_downstream_batch(_array(c["wq"]), _array(dq_rows))
        + layer_downstream_batch(_array(c["wk"]), _array(dk_rows))
    ) + layer_downstream_batch(_array(c["wv"]), _array(dv_rows))
    return {"dx": _numpy(dx).reshape(c["x"].shape), "dq": dq_rows, "dk": dk_rows, "dv": dv_rows}


def _backward(c: dict[str, np.ndarray], f: dict[str, np.ndarray], heads: int) -> dict[str, np.ndarray]:
    result = attention_downstream_batch(
        _array(c["delta"]),
        *(_array(c[name]) for name in ("wq", "wk", "wv", "wo")),
        *(_array(f[name]) for name in ("q", "k", "v", "p")),
        heads=heads,
    )
    return {name: _numpy(array) for name, array in zip(["dx", "dq", "dk", "dv"], result)}


def _gradients(c: dict[str, np.ndarray], f: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[np.ndarray]:
    grads = attention_accumulate_gradient_batch(
        _array(c["delta"]),
        _array(c["x"]),
        _array(f["h"]),
        *(_array(b[name]) for name in ("dq", "dk", "dv")),
        *(_array(c["grad_" + name]) for name in NAMES),
    )
    return [_numpy(g) for g in grads]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_the_forward_pass_is_the_crates_composition(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    # project (q, k, v), attend (p, h) and combine (a), each block's outputs
    c = _case(examples, tokens, features, heads, key_size)
    out, composed = _forward(c, heads, causal), _composed_forward(c, tokens, heads, causal)
    for name in ["a", "q", "k", "v", "p", "h"]:
        assert out[name].shape == composed[name].shape, name
        assert _bits(out[name]) == _bits(composed[name]), name


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_one_examples_forward_pass_is_its_batch_rows(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    c = _case(examples, tokens, features, heads, key_size)
    batch = _forward(c, heads, causal)
    for i in range(examples):
        result = attention_forward(_array(c["x"][i]), *_parameters(c), heads=heads, causal=causal)
        single = {name: _numpy(array) for name, array in zip(["a", "q", "k", "v", "p", "h"], result)}
        assert _bits(single["a"]) == _bits(batch["a"][i])
        for name in ["q", "k", "v", "p", "h"]:
            assert single[name].shape == (tokens, batch[name].shape[1]), name
            assert _bits(single[name]) == _bits(batch[name][i * tokens : (i + 1) * tokens]), name


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_the_downstream_is_the_crates_composition(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    # combine's and attend's backward (dq, dk, dv), then project's (dx)
    c = _case(examples, tokens, features, heads, key_size)
    f = _forward(c, heads, causal)
    out, composed = _backward(c, f, heads), _composed_backward(c, f, tokens, heads)
    for name in ["dx", "dq", "dk", "dv"]:
        assert out[name].shape == composed[name].shape, name
        assert _bits(out[name]) == _bits(composed[name]), name


@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_the_gradients_are_each_projections_dense_gradients(
    examples: int, tokens: int, features: int, heads: int, key_size: int
):
    c = _case(examples, tokens, features, heads, key_size)
    f = _forward(c, heads)
    b = _backward(c, f, heads)
    grads = _gradients(c, f, b)
    rows, delta = _array(c["x"].reshape(-1, features)), _array(c["delta"].reshape(-1, features))
    inputs = {"q": (b["dq"], rows), "k": (b["dk"], rows), "v": (b["dv"], rows), "o": (None, _array(f["h"]))}
    for i, n in enumerate("qkvo"):
        d_rows, x_rows = inputs[n]
        expected = layer_accumulate_gradient_batch(
            delta if d_rows is None else _array(d_rows), x_rows, _array(c["grad_w" + n]), _array(c["grad_b" + n])
        )
        assert _bits(grads[2 * i]) == _bits(_numpy(expected[0])), "w" + n
        assert _bits(grads[2 * i + 1]) == _bits(_numpy(expected[1])), "b" + n


def _digest(arrays: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for values in arrays:
        h.update(repr(values.shape).encode())
        h.update(values.astype(np.float64).tobytes())
    return h.hexdigest()[:16]


def _digests(c: dict[str, np.ndarray], x: np.ndarray, delta: np.ndarray, one: bool) -> list[str]:
    forward = attention_forward if one else attention_forward_batch
    f = dict(zip(["a", "q", "k", "v", "p", "h"], (_numpy(a) for a in forward(_array(x), *_parameters(c), heads=1))))
    b = _backward({**c, "delta": delta}, f, 1)
    grads = _gradients({**c, "x": x, "delta": delta}, f, b)
    return [_digest(list(f.values())), _digest(list(b.values())), _digest(grads)]


@pytest.mark.parametrize("examples,tokens,features,heads,key_size", ONE_HEAD, ids=map(_id, ONE_HEAD))
def test_at_one_head_the_ops_are_the_single_head_ops_recorded_before_the_change(
    examples: int, tokens: int, features: int, heads: int, key_size: int
):
    c = _case(examples, tokens, features, heads, key_size)
    batch = _digests(c, c["x"], c["delta"], one=False)
    one = _digests(c, c["x"][0], c["delta"][0], one=True)
    assert tuple(batch + one) == RECORDED[(examples, tokens, features)]


@pytest.mark.parametrize("examples,features,heads,key_size", [(1, 1, 1, 1), (3, 5, 1, 5), (2, 32, 4, 8), (2, 6, 3, 5)])
def test_with_one_token_attention_is_the_value_then_the_output_projection(
    examples: int, features: int, heads: int, key_size: int
):
    # every head's P[i] = [[1]], so out = (X Wv^T + bv) Wo^T + bo by bits
    c = _case(examples, 1, features, heads, key_size)
    out = _forward(c, heads)
    assert out["p"].shape == (examples, heads)
    assert np.all(out["p"] == 1.0)
    v = affine_forward_batch(_array(c["wv"]), _array(c["x"]), _array(c["bv"]))
    expected = affine_forward_batch(_array(c["wo"]), v, _array(c["bo"]))
    assert _bits(out["a"]) == _bits(_numpy(expected))


@pytest.mark.parametrize(
    "examples,features,heads,key_size", [(1, 1, 1, 1), (2, 7, 1, 7), (2, 32, 1, 32), (2, 32, 4, 8)]
)
def test_with_zero_query_and_key_weights_attention_is_uniform_and_h_the_token_mean_of_v(
    examples: int, features: int, heads: int, key_size: int
):
    # every score is 0 and every weight exactly 1/16, so each row of H[i] is the mean of V[i]'s
    # rows, 1/16 * v exact and the product's sum a left fold, as the token mean's
    tokens = 16
    c = _case(examples, tokens, features, heads, key_size)
    for name in ("wq", "bq", "wk", "bk"):
        c[name] = np.zeros_like(c[name])
    out = _forward(c, heads)
    assert np.all(out["p"] == 1.0 / 16)
    for n in range(examples):
        v_n = out["v"][n * tokens : (n + 1) * tokens]
        mean = _numpy(token_mean_forward(_array(v_n.ravel()), tokens))
        for row in out["h"][n * tokens : (n + 1) * tokens]:
            assert _bits(row) == _bits(mean)


def _repeated(block: np.ndarray, heads: int) -> np.ndarray:
    # one head's rows of a weight or bias, as every head's
    return np.concatenate([block] * heads)


@pytest.mark.parametrize("examples,tokens,features,heads,key_size", MULTI_HEAD, ids=map(_id, MULTI_HEAD))
def test_identical_heads_have_identical_weights_and_outputs(
    examples: int, tokens: int, features: int, heads: int, key_size: int
):
    c = _case(examples, tokens, features, heads, key_size)
    for name in ("wq", "bq", "wk", "bk", "wv", "bv"):
        c[name] = _repeated(c[name][:key_size], heads)
    out = _forward(c, heads)
    for n in range(examples):
        for i in range(1, heads):
            assert _bits(_head(out["p"], n, i, tokens, tokens)) == _bits(_head(out["p"], n, 0, tokens, tokens))
            assert _bits(_head(out["h"], n, i, tokens, key_size)) == _bits(_head(out["h"], n, 0, tokens, key_size))


@pytest.mark.parametrize("examples,tokens,features,heads,key_size", MULTI_HEAD, ids=map(_id, MULTI_HEAD))
def test_a_silent_heads_gradients_are_zero(examples: int, tokens: int, features: int, heads: int, key_size: int):
    # head 0's columns of Wo zero: dH[0] is exactly zero, so are its blocks of dq, dk, dv and of
    # the gradients of Wq, Wk, Wv and their biases (accumulated from zero)
    silent = slice(0, key_size)
    c = _case(examples, tokens, features, heads, key_size)
    c["wo"][:, silent] = 0.0
    for name in NAMES:
        c["grad_" + name] = np.zeros_like(c[name])
    f = _forward(c, heads)
    b = _backward(c, f, heads)
    for name in ("dq", "dk", "dv"):
        assert _bits(b[name][:, silent]) == _bits(np.zeros_like(b[name][:, silent])), name
        assert np.any(b[name][:, key_size:] != 0.0), name
    grads = dict(zip(NAMES, _gradients(c, f, b)))
    for name in ("wq", "bq", "wk", "bk", "wv", "bv"):
        assert _bits(grads[name][silent]) == _bits(np.zeros_like(grads[name][silent])), name


@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_a_causal_pass_weights_no_later_token_and_no_later_token_moves_an_earlier_output(
    examples: int, tokens: int, features: int, heads: int, key_size: int
):
    # every masked weight P_tj, j > t, exactly 0; and token t's outputs, its rows of a and h,
    # read only tokens 0..t, so new values for the later tokens leave them unchanged by bits
    c = _case(examples, tokens, features, heads, key_size)
    out = _forward(c, heads, causal=True)
    future = np.triu(np.ones((tokens, tokens), dtype=np.bool_), k=1)
    for n in range(examples):
        for i in range(heads):
            p = _head(out["p"], n, i, tokens, tokens)
            assert _bits(p[future]) == _bits(np.zeros(int(future.sum())))
            assert np.all(p[~future] > 0.0)
    for t in range(tokens - 1):
        later = dict(c)
        later["x"] = c["x"].copy()
        later["x"][:, (t + 1) * features :] = np.random.default_rng(t).uniform(
            -2.0, 2.0, (examples, (tokens - t - 1) * features)
        )
        moved = _forward(later, heads, causal=True)
        a, a_moved = out["a"].reshape(examples, tokens, features), moved["a"].reshape(examples, tokens, features)
        assert _bits(a[:, : t + 1]) == _bits(a_moved[:, : t + 1]), t
        h, h_moved = out["h"].reshape(examples, tokens, -1), moved["h"].reshape(examples, tokens, -1)
        assert _bits(h[:, : t + 1]) == _bits(h_moved[:, : t + 1]), t


@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_unmasked_is_the_default_and_the_first_tokens_row_is_its_value_projection_masked(
    examples: int, tokens: int, features: int, heads: int, key_size: int
):
    c = _case(examples, tokens, features, heads, key_size)
    default, unmasked, masked = _forward(c, heads), _forward(c, heads, causal=False), _forward(c, heads, causal=True)
    for name in ["a", "q", "k", "v", "p", "h"]:
        assert _bits(default[name]) == _bits(unmasked[name]), name
    # token 0 weights only itself, P_00 = 1, so its row of H is its row of V exactly
    for n in range(examples):
        assert _bits(masked["h"][n * tokens]) == _bits(masked["v"][n * tokens])
    if tokens == 1:
        for name in ["a", "q", "k", "v", "p", "h"]:
            assert _bits(masked[name]) == _bits(unmasked[name]), name


def test_the_attention_ops_refuse_shapes_that_dont_fit():
    c = _case(1, 2, 6, 2, 2)
    parameters = _parameters(c)
    with pytest.raises(ValueError, match=r"attention_forward_batch requires x of whole tokens of 6 features"):
        attention_forward_batch(Array([[0.0] * 4]), *parameters, heads=2)
    with pytest.raises(ValueError, match="attention_forward_batch requires a 2D x"):
        attention_forward_batch(Array([0.0] * 12), *parameters, heads=2)
    with pytest.raises(ValueError, match="attention_forward requires a 1D x"):
        attention_forward(Array([[0.0] * 12]), *parameters, heads=2)
    for heads in (0, 3):
        with pytest.raises(ValueError, match=rf"heads >= 1 dividing the projections' width 4, got heads={heads}"):
            attention_forward(Array([0.0] * 12), *parameters, heads=heads)
    wq, bq, wk, bk, wv, bv, wo, bo = parameters
    with pytest.raises(ValueError, match=r"requires wo of shape Matrix\(6, 4\)"):
        attention_forward(Array([0.0] * 12), wq, bq, wk, bk, wv, bv, wq, bo, 2)
    with pytest.raises(ValueError, match=r"requires bk of shape Vector\(4\)"):
        attention_forward(Array([0.0] * 12), wq, bq, wk, bo, wv, bv, wo, bo, 2)
    f = _forward(c, 2)
    q, k, v, p = (_array(f[name]) for name in ("q", "k", "v", "p"))
    with pytest.raises(ValueError, match=r"attention_downstream_batch requires p of shape Matrix\(2, 4\)"):
        attention_downstream_batch(_array(c["delta"]), wq, wk, wv, wo, q, k, v, _array(f["p"][:, :2]), 2)
    with pytest.raises(ValueError, match=r"attention_downstream_batch requires wo of shape Matrix\(6, 4\)"):
        attention_downstream_batch(_array(c["delta"]), wq, wk, wv, wq, q, k, v, p, 2)


@pytest.mark.parametrize(
    "tokens,features,heads,key_size", [(1, 3, 1, 3), (5, 6, 1, 6), (16, 32, 1, 32), (5, 6, 3, 2), (16, 32, 4, 5)]
)
def test_one_examples_backward_is_a_batch_of_ones(tokens: int, features: int, heads: int, key_size: int):
    c = _case(1, tokens, features, heads, key_size)
    x, delta = c["x"][0], c["delta"][0]
    _, q, k, v, p, h = attention_forward(_array(x), *_parameters(c), heads=heads)
    weights = [_array(c[name]) for name in ("wq", "wk", "wv", "wo")]
    single = attention_downstream_batch(
        _array(delta), weights[0], weights[1], weights[2], weights[3], q, k, v, p, heads
    )
    f = _forward(c, heads)
    batch = _backward(c, f, heads)
    assert single[0].shape == (tokens * features,)
    assert _bits(_numpy(single[0])) == _bits(batch["dx"])
    for array, name in zip(single[1:], ["dq", "dk", "dv"]):
        assert _bits(_numpy(array)) == _bits(batch[name]), name
    grads = [_array(c["grad_" + name]) for name in NAMES]
    one = attention_accumulate_gradient_batch(_array(delta), _array(x), h, single[1], single[2], single[3], *grads)
    many = _gradients(c, f, batch)
    for i, name in enumerate(NAMES):
        assert _bits(_numpy(one[i])) == _bits(many[i]), name
