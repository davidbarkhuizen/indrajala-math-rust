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

Dropout on the weights (indrajala-ml's attention-dropout workplan, D3-D5): the mask is numpy's
`default_rng(seed).random((N, h, T, T)) >= dropout`, packed as `p`, and the passes are the
composition with `P~ = P * M / keep` (numpy, exact) in place of `P` in `H` and `dV`, and `dP`
times `M / keep`. Without training, and at dropout 0, nothing is drawn and the bits are as before.
"""

import hashlib
import math
from typing import Any

import numpy as np
import pytest
from array_helpers import bits, to_array, to_numpy

from indrajala_math_rust import (
    Array,
    Generator,
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
    return [to_array(c[name]) for name in NAMES]


def _head(rows: np.ndarray, n: int, i: int, tokens: int, width: int) -> np.ndarray:
    # example n's head i: rows n * T.., columns i * width..
    return rows[n * tokens : (n + 1) * tokens, i * width : (i + 1) * width]


def _side_by_side(blocks: list[list[np.ndarray]]) -> np.ndarray:
    # blocks[n][i], each (T, width), as (N * T, h * width) rows
    return np.concatenate([np.concatenate(heads, axis=1) for heads in blocks])


def _masked(scores: Array, tokens: int) -> Array:
    # each S_tj with j > t set to -inf
    future = np.triu(np.ones((tokens, tokens), dtype=np.bool_), k=1)
    return to_array(np.where(future, -np.inf, to_numpy(scores)))


def _composed_forward(c: dict[str, np.ndarray], tokens: int, heads: int, causal: bool = False) -> dict[str, np.ndarray]:
    d, width = c["bo"].shape[0], c["bq"].shape[0]
    d_k = width // heads
    rows = to_array(c["x"].reshape(-1, d))
    q, k, v = (to_numpy(affine_forward_batch(to_array(c["w" + n]), rows, to_array(c["b" + n]))) for n in "qkv")
    p: list[list[np.ndarray]] = []
    h: list[list[np.ndarray]] = []
    for n in range(c["x"].shape[0]):
        p.append([])
        h.append([])
        for i in range(heads):
            q_ni, k_ni, v_ni = (to_array(_head(m, n, i, tokens, d_k)) for m in (q, k, v))
            scores = (q_ni @ k_ni.T) / math.sqrt(d_k)
            p_ni = array_softmax(_masked(scores, tokens) if causal else scores)
            p[n].append(to_numpy(p_ni))
            h[n].append(to_numpy(p_ni @ v_ni))
    h_rows = _side_by_side(h)
    a = to_numpy(affine_forward_batch(to_array(c["wo"]), to_array(h_rows), to_array(c["bo"])))
    return {"a": a.reshape(c["x"].shape), "q": q, "k": k, "v": v, "p": _side_by_side(p), "h": h_rows}


def _forward(c: dict[str, np.ndarray], heads: int, causal: bool = False) -> dict[str, np.ndarray]:
    result = attention_forward_batch(to_array(c["x"]), *_parameters(c), heads=heads, causal=causal)
    return {name: to_numpy(array) for name, array in zip(["a", "q", "k", "v", "p", "h"], result[:6])}


def _composed_backward(
    c: dict[str, np.ndarray], f: dict[str, np.ndarray], tokens: int, heads: int
) -> dict[str, np.ndarray]:
    d, width = c["bo"].shape[0], c["bq"].shape[0]
    d_k = width // heads
    s = math.sqrt(d_k)
    dh = to_numpy(layer_downstream_batch(to_array(c["wo"]), to_array(c["delta"].reshape(-1, d))))
    dq: list[list[np.ndarray]] = []
    dk: list[list[np.ndarray]] = []
    dv: list[list[np.ndarray]] = []
    for n in range(c["x"].shape[0]):
        for blocks in (dq, dk, dv):
            blocks.append([])
        for i in range(heads):
            q_ni, k_ni, v_ni, dh_ni = (to_array(_head(m, n, i, tokens, d_k)) for m in (f["q"], f["k"], f["v"], dh))
            p = _head(f["p"], n, i, tokens, tokens)
            dp = to_numpy(dh_ni @ v_ni.T)
            dv[n].append(to_numpy(to_array(p).T @ dh_ni))
            r = to_numpy(sum_axis0(to_array((dp * p).T)))  # each row's left fold
            ds = to_array(p * (dp - r[:, np.newaxis]))
            dq[n].append(to_numpy((ds @ k_ni) / s))
            dk[n].append(to_numpy((ds.T @ q_ni) / s))
    dq_rows, dk_rows, dv_rows = _side_by_side(dq), _side_by_side(dk), _side_by_side(dv)
    dx = (
        layer_downstream_batch(to_array(c["wq"]), to_array(dq_rows))
        + layer_downstream_batch(to_array(c["wk"]), to_array(dk_rows))
    ) + layer_downstream_batch(to_array(c["wv"]), to_array(dv_rows))
    return {"dx": to_numpy(dx).reshape(c["x"].shape), "dq": dq_rows, "dk": dk_rows, "dv": dv_rows}


def _backward(c: dict[str, np.ndarray], f: dict[str, np.ndarray], heads: int) -> dict[str, np.ndarray]:
    result = attention_downstream_batch(
        to_array(c["delta"]),
        *(to_array(c[name]) for name in ("wq", "wk", "wv", "wo")),
        *(to_array(f[name]) for name in ("q", "k", "v", "p")),
        heads=heads,
    )
    return {name: to_numpy(array) for name, array in zip(["dx", "dq", "dk", "dv"], result)}


def _gradients(c: dict[str, np.ndarray], f: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[np.ndarray]:
    grads = attention_accumulate_gradient_batch(
        to_array(c["delta"]),
        to_array(c["x"]),
        to_array(f["h"]),
        *(to_array(b[name]) for name in ("dq", "dk", "dv")),
        *(to_array(c["grad_" + name]) for name in NAMES),
    )
    return [to_numpy(g) for g in grads]


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
        assert bits(out[name]) == bits(composed[name]), name


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_one_examples_forward_pass_is_its_batch_rows(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    c = _case(examples, tokens, features, heads, key_size)
    batch = _forward(c, heads, causal)
    for i in range(examples):
        result = attention_forward(to_array(c["x"][i]), *_parameters(c), heads=heads, causal=causal)
        single = {name: to_numpy(array) for name, array in zip(["a", "q", "k", "v", "p", "h"], result[:6])}
        assert bits(single["a"]) == bits(batch["a"][i])
        for name in ["q", "k", "v", "p", "h"]:
            assert single[name].shape == (tokens, batch[name].shape[1]), name
            assert bits(single[name]) == bits(batch[name][i * tokens : (i + 1) * tokens]), name


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
        assert bits(out[name]) == bits(composed[name]), name


@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_the_gradients_are_each_projections_dense_gradients(
    examples: int, tokens: int, features: int, heads: int, key_size: int
):
    c = _case(examples, tokens, features, heads, key_size)
    f = _forward(c, heads)
    b = _backward(c, f, heads)
    grads = _gradients(c, f, b)
    rows, delta = to_array(c["x"].reshape(-1, features)), to_array(c["delta"].reshape(-1, features))
    inputs = {"q": (b["dq"], rows), "k": (b["dk"], rows), "v": (b["dv"], rows), "o": (None, to_array(f["h"]))}
    for i, n in enumerate("qkvo"):
        d_rows, x_rows = inputs[n]
        expected = layer_accumulate_gradient_batch(
            delta if d_rows is None else to_array(d_rows), x_rows, to_array(c["grad_w" + n]), to_array(c["grad_b" + n])
        )
        assert bits(grads[2 * i]) == bits(to_numpy(expected[0])), "w" + n
        assert bits(grads[2 * i + 1]) == bits(to_numpy(expected[1])), "b" + n


def _digest(arrays: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for values in arrays:
        h.update(repr(values.shape).encode())
        h.update(values.astype(np.float64).tobytes())
    return h.hexdigest()[:16]


def _digests(c: dict[str, np.ndarray], x: np.ndarray, delta: np.ndarray, one: bool) -> list[str]:
    forward = attention_forward if one else attention_forward_batch
    f = dict(
        zip(["a", "q", "k", "v", "p", "h"], (to_numpy(a) for a in forward(to_array(x), *_parameters(c), heads=1)[:6]))
    )
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
    v = affine_forward_batch(to_array(c["wv"]), to_array(c["x"]), to_array(c["bv"]))
    expected = affine_forward_batch(to_array(c["wo"]), v, to_array(c["bo"]))
    assert bits(out["a"]) == bits(to_numpy(expected))


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
        mean = to_numpy(token_mean_forward(to_array(v_n.ravel()), tokens))
        for row in out["h"][n * tokens : (n + 1) * tokens]:
            assert bits(row) == bits(mean)


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
            assert bits(_head(out["p"], n, i, tokens, tokens)) == bits(_head(out["p"], n, 0, tokens, tokens))
            assert bits(_head(out["h"], n, i, tokens, key_size)) == bits(_head(out["h"], n, 0, tokens, key_size))


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
        assert bits(b[name][:, silent]) == bits(np.zeros_like(b[name][:, silent])), name
        assert np.any(b[name][:, key_size:] != 0.0), name
    grads = dict(zip(NAMES, _gradients(c, f, b)))
    for name in ("wq", "bq", "wk", "bk", "wv", "bv"):
        assert bits(grads[name][silent]) == bits(np.zeros_like(grads[name][silent])), name


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
            assert bits(p[future]) == bits(np.zeros(int(future.sum())))
            assert np.all(p[~future] > 0.0)
    for t in range(tokens - 1):
        later = dict(c)
        later["x"] = c["x"].copy()
        later["x"][:, (t + 1) * features :] = np.random.default_rng(t).uniform(
            -2.0, 2.0, (examples, (tokens - t - 1) * features)
        )
        moved = _forward(later, heads, causal=True)
        a, a_moved = out["a"].reshape(examples, tokens, features), moved["a"].reshape(examples, tokens, features)
        assert bits(a[:, : t + 1]) == bits(a_moved[:, : t + 1]), t
        h, h_moved = out["h"].reshape(examples, tokens, -1), moved["h"].reshape(examples, tokens, -1)
        assert bits(h[:, : t + 1]) == bits(h_moved[:, : t + 1]), t


@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_unmasked_is_the_default_and_the_first_tokens_row_is_its_value_projection_masked(
    examples: int, tokens: int, features: int, heads: int, key_size: int
):
    c = _case(examples, tokens, features, heads, key_size)
    default, unmasked, masked = _forward(c, heads), _forward(c, heads, causal=False), _forward(c, heads, causal=True)
    for name in ["a", "q", "k", "v", "p", "h"]:
        assert bits(default[name]) == bits(unmasked[name]), name
    # token 0 weights only itself, P_00 = 1, so its row of H is its row of V exactly
    for n in range(examples):
        assert bits(masked["h"][n * tokens]) == bits(masked["v"][n * tokens])
    if tokens == 1:
        for name in ["a", "q", "k", "v", "p", "h"]:
            assert bits(masked[name]) == bits(unmasked[name]), name


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
    q, k, v, p = (to_array(f[name]) for name in ("q", "k", "v", "p"))
    with pytest.raises(ValueError, match=r"attention_downstream_batch requires p of shape Matrix\(2, 4\)"):
        attention_downstream_batch(to_array(c["delta"]), wq, wk, wv, wo, q, k, v, to_array(f["p"][:, :2]), 2)
    with pytest.raises(ValueError, match=r"attention_downstream_batch requires wo of shape Matrix\(6, 4\)"):
        attention_downstream_batch(to_array(c["delta"]), wq, wk, wv, wq, q, k, v, p, 2)


@pytest.mark.parametrize(
    "tokens,features,heads,key_size", [(1, 3, 1, 3), (5, 6, 1, 6), (16, 32, 1, 32), (5, 6, 3, 2), (16, 32, 4, 5)]
)
def test_one_examples_backward_is_a_batch_of_ones(tokens: int, features: int, heads: int, key_size: int):
    c = _case(1, tokens, features, heads, key_size)
    x, delta = c["x"][0], c["delta"][0]
    _, q, k, v, p, h, _ = attention_forward(to_array(x), *_parameters(c), heads=heads)
    weights = [to_array(c[name]) for name in ("wq", "wk", "wv", "wo")]
    single = attention_downstream_batch(
        to_array(delta), weights[0], weights[1], weights[2], weights[3], q, k, v, p, heads
    )
    f = _forward(c, heads)
    batch = _backward(c, f, heads)
    assert single[0].shape == (tokens * features,)
    assert bits(to_numpy(single[0])) == bits(batch["dx"])
    for array, name in zip(single[1:], ["dq", "dk", "dv"]):
        assert bits(to_numpy(array)) == bits(batch[name]), name
    grads = [to_array(c["grad_" + name]) for name in NAMES]
    one = attention_accumulate_gradient_batch(to_array(delta), to_array(x), h, single[1], single[2], single[3], *grads)
    many = _gradients(c, f, batch)
    for i, name in enumerate(NAMES):
        assert bits(to_numpy(one[i])) == bits(many[i]), name


# dropout on the weights (the attention-dropout workplan, stage 4)

DROPOUT = 0.3
FORWARD = ["a", "q", "k", "v", "p", "h", "mask"]


def _dropping_forward(
    c: dict[str, np.ndarray], heads: int, causal: bool, seed: int, one: bool = False
) -> dict[str, Any]:
    forward = attention_forward if one else attention_forward_batch
    x = c["x"][0] if one else c["x"]
    rng = Generator(seed)
    result = forward(to_array(x), *_parameters(c), heads=heads, causal=causal, dropout=DROPOUT, training=True, rng=rng)
    out: dict[str, Any] = {name: None if array is None else to_numpy(array) for name, array in zip(FORWARD, result)}
    out["rng"] = rng.state
    return out


def _packed_mask(examples: int, tokens: int, heads: int, seed: int) -> np.ndarray:
    # numpy's (N, h, T, T) draw, packed as p: (N * T, h * T), head i in columns i * T..
    mask = (np.random.default_rng(seed).random((examples, heads, tokens, tokens)) >= DROPOUT).astype(np.float64)
    return mask.transpose(0, 2, 1, 3).reshape(examples * tokens, heads * tokens)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_without_training_or_at_dropout_0_nothing_is_drawn_and_the_bits_are_as_before(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    c = _case(examples, tokens, features, heads, key_size)
    before = _forward(c, heads, causal)
    for dropout, training in [(DROPOUT, False), (0.0, True)]:
        rng = Generator(4)
        state = rng.state
        result = attention_forward_batch(
            to_array(c["x"]), *_parameters(c), heads=heads, causal=causal, dropout=dropout, training=training, rng=rng
        )
        assert result[6] is None and rng.state == state
        for name, array in zip(FORWARD, result[:6]):
            assert bits(to_numpy(array)) == bits(before[name]), name
        b = attention_downstream_batch(
            to_array(c["delta"]),
            *(to_array(c[name]) for name in ("wq", "wk", "wv", "wo")),
            *(to_array(before[name]) for name in ("q", "k", "v", "p")),
            heads=heads,
            mask=None,
            dropout=dropout,
        )
        for name, array in zip(["dx", "dq", "dk", "dv"], b):
            assert bits(to_numpy(array)) == bits(_backward(c, before, heads)[name]), name


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_the_mask_is_numpys_row_major_draw_every_entry_drawn_packed_as_p(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    c = _case(examples, tokens, features, heads, key_size)
    out = _dropping_forward(c, heads, causal, seed=7)
    assert bits(out["mask"]) == bits(_packed_mask(examples, tokens, heads, 7))
    # exactly N * h * T * T draws: the generator is where numpy's one draw leaves it
    after = np.random.default_rng(7)
    after.random(examples * heads * tokens * tokens)
    assert out["rng"] == after.bit_generator.state


def _dropped_composition(
    c: dict[str, np.ndarray], f: dict[str, np.ndarray], tokens: int, heads: int
) -> dict[str, np.ndarray]:
    # the forward pass's h and a, and the backward pass, with P~ = P * M / keep (numpy, exact)
    d, width = c["bo"].shape[0], c["bq"].shape[0]
    d_k, s, keep = width // heads, math.sqrt(width // heads), 1.0 - DROPOUT
    dh = to_numpy(layer_downstream_batch(to_array(c["wo"]), to_array(c["delta"].reshape(-1, d))))
    h: list[list[np.ndarray]] = []
    dq: list[list[np.ndarray]] = []
    dk: list[list[np.ndarray]] = []
    dv: list[list[np.ndarray]] = []
    for n in range(c["x"].shape[0]):
        for blocks in (h, dq, dk, dv):
            blocks.append([])
        for i in range(heads):
            q_ni, k_ni, v_ni, dh_ni = (to_array(_head(m, n, i, tokens, d_k)) for m in (f["q"], f["k"], f["v"], dh))
            p, m = _head(f["p"], n, i, tokens, tokens), _head(f["mask"], n, i, tokens, tokens)
            dropped = to_array(p * m / keep)
            h[n].append(to_numpy(dropped @ v_ni))
            dp = to_numpy(dh_ni @ v_ni.T) * m / keep
            dv[n].append(to_numpy(dropped.T @ dh_ni))
            r = to_numpy(sum_axis0(to_array((dp * p).T)))
            ds = to_array(p * (dp - r[:, np.newaxis]))
            dq[n].append(to_numpy((ds @ k_ni) / s))
            dk[n].append(to_numpy((ds.T @ q_ni) / s))
    h_rows = _side_by_side(h)
    a = to_numpy(affine_forward_batch(to_array(c["wo"]), to_array(h_rows), to_array(c["bo"])))
    dq_rows, dk_rows, dv_rows = _side_by_side(dq), _side_by_side(dk), _side_by_side(dv)
    dx = (
        layer_downstream_batch(to_array(c["wq"]), to_array(dq_rows))
        + layer_downstream_batch(to_array(c["wk"]), to_array(dk_rows))
    ) + layer_downstream_batch(to_array(c["wv"]), to_array(dv_rows))
    return {
        "a": a.reshape(c["x"].shape),
        "h": h_rows,
        "dx": to_numpy(dx).reshape(c["x"].shape),
        "dq": dq_rows,
        "dk": dk_rows,
        "dv": dv_rows,
    }


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_a_dropping_pass_is_the_composition_with_p_tilde(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    c = _case(examples, tokens, features, heads, key_size)
    out = _dropping_forward(c, heads, causal, seed=7)
    plain = _forward(c, heads, causal)
    # project and the softmax as without dropout; attend's H and combine's a from P~
    for name in ["q", "k", "v", "p"]:
        assert bits(out[name]) == bits(plain[name]), name
    composed = _dropped_composition(c, out, tokens, heads)
    for name in ["a", "h"]:
        assert bits(out[name]) == bits(composed[name]), name
    b = attention_downstream_batch(
        to_array(c["delta"]),
        *(to_array(c[name]) for name in ("wq", "wk", "wv", "wo")),
        *(to_array(out[name]) for name in ("q", "k", "v", "p")),
        heads=heads,
        mask=to_array(out["mask"]),
        dropout=DROPOUT,
    )
    for name, array in zip(["dx", "dq", "dk", "dv"], b):
        assert bits(to_numpy(array)) == bits(composed[name]), name


@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_one_examples_dropping_pass_is_its_batch_of_ones(
    examples: int, tokens: int, features: int, heads: int, key_size: int
):
    c = _case(1, tokens, features, heads, key_size)
    one, batch = _dropping_forward(c, heads, True, seed=3, one=True), _dropping_forward(c, heads, True, seed=3)
    assert bits(one["a"]) == bits(batch["a"][0])
    for name in ["p", "h", "mask"]:
        assert bits(one[name]) == bits(batch[name]), name
    assert one["rng"] == batch["rng"]


def test_the_dropping_ops_refuse_a_dropout_outside_0_to_1_a_missing_rng_and_a_misshapen_mask():
    c = _case(2, 3, 4, 2, 2)
    x = to_array(c["x"])
    for dropout in (-0.1, 1.0, math.nan):
        with pytest.raises(ValueError, match=r"attention_forward_batch requires a dropout in \[0, 1\)"):
            attention_forward_batch(x, *_parameters(c), heads=2, dropout=dropout)
    with pytest.raises(ValueError, match="attention_forward requires an rng to drop out in training"):
        attention_forward(to_array(c["x"][0]), *_parameters(c), heads=2, dropout=0.1, training=True)
    f = _dropping_forward(c, 2, False, seed=1)
    args = (
        to_array(c["delta"]),
        *(to_array(c[name]) for name in ("wq", "wk", "wv", "wo")),
        *(to_array(f[name]) for name in ("q", "k", "v", "p")),
    )
    with pytest.raises(ValueError, match=r"attention_downstream_batch requires mask of shape Matrix\(6, 6\)"):
        attention_downstream_batch(*args, heads=2, mask=to_array(f["mask"][:, :3]), dropout=DROPOUT)
    with pytest.raises(ValueError, match=r"attention_downstream_batch requires a dropout in \[0, 1\)"):
        attention_downstream_batch(*args, heads=2, mask=to_array(f["mask"]), dropout=1.0)


# Rotary positions (indrajala-ml's rotary-positions workplan, D3-D5): the forward ops given `cos`
# and `sin` are the composition with Q and K rotated by numpy, exact, after project; the downstream
# is the composition from the rotated caches, its dq and dk rotated back before project's products.

ROTARY = [shape for shape in SHAPES if shape[4] % 2 == 0]


def _table(tokens: int, key_size: int, base: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    # indrajala-ml's rotary_table: math's cos and sin of t * theta_j, (T, d_k / 2) each
    angles = [[t * base ** (-2 * j / key_size) for j in range(key_size // 2)] for t in range(tokens)]
    return (
        np.array([[math.cos(a) for a in row] for row in angles]).reshape(tokens, key_size // 2),
        np.array([[math.sin(a) for a in row] for row in angles]).reshape(tokens, key_size // 2),
    )


def _rotated(rows: np.ndarray, table: tuple[np.ndarray, np.ndarray], heads: int, inverse: bool = False) -> np.ndarray:
    # (N * T, h * d_k) rows, each head's features j and j + d_k / 2 rotated by the row's token's angles
    tokens, half = table[0].shape
    x = rows.reshape(-1, tokens, heads, 2 * half)
    a, b = x[..., :half], x[..., half:]
    c, s = table[0][:, np.newaxis, :], table[1][:, np.newaxis, :]
    pairs = (a * c + b * s, b * c - a * s) if inverse else (a * c - b * s, a * s + b * c)
    return np.concatenate(pairs, axis=-1).reshape(rows.shape)


def _rotary_forward(
    c: dict[str, np.ndarray], tokens: int, heads: int, causal: bool, one: bool = False
) -> dict[str, np.ndarray]:
    cos, sin = _table(tokens, c["bq"].shape[0] // heads)
    x, forward = (c["x"][0], attention_forward) if one else (c["x"], attention_forward_batch)
    result = forward(to_array(x), *_parameters(c), heads=heads, causal=causal, cos=to_array(cos), sin=to_array(sin))
    return {name: to_numpy(array) for name, array in zip(["a", "q", "k", "v", "p", "h"], result[:6])}


def _composed_rotary_forward(c: dict[str, np.ndarray], tokens: int, heads: int, causal: bool) -> dict[str, np.ndarray]:
    # project, then Q and K rotated, then attend and combine as _composed_forward's
    d, width = c["bo"].shape[0], c["bq"].shape[0]
    d_k = width // heads
    table = _table(tokens, d_k)
    rows = to_array(c["x"].reshape(-1, d))
    q, k, v = (to_numpy(affine_forward_batch(to_array(c["w" + n]), rows, to_array(c["b" + n]))) for n in "qkv")
    q, k = _rotated(q, table, heads), _rotated(k, table, heads)
    p: list[list[np.ndarray]] = []
    h: list[list[np.ndarray]] = []
    for n in range(c["x"].shape[0]):
        p.append([])
        h.append([])
        for i in range(heads):
            q_ni, k_ni, v_ni = (to_array(_head(m, n, i, tokens, d_k)) for m in (q, k, v))
            scores = (q_ni @ k_ni.T) / math.sqrt(d_k)
            p_ni = array_softmax(_masked(scores, tokens) if causal else scores)
            p[n].append(to_numpy(p_ni))
            h[n].append(to_numpy(p_ni @ v_ni))
    h_rows = _side_by_side(h)
    a = to_numpy(affine_forward_batch(to_array(c["wo"]), to_array(h_rows), to_array(c["bo"])))
    return {"a": a.reshape(c["x"].shape), "q": q, "k": k, "v": v, "p": _side_by_side(p), "h": h_rows}


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", ROTARY, ids=map(_id, ROTARY))
def test_a_rotary_forward_pass_is_the_composition_with_q_and_k_rotated_after_project(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    c = _case(examples, tokens, features, heads, key_size)
    out, composed = _rotary_forward(c, tokens, heads, causal), _composed_rotary_forward(c, tokens, heads, causal)
    for name in ["a", "q", "k", "v", "p", "h"]:
        assert out[name].shape == composed[name].shape, name
        assert bits(out[name]) == bits(composed[name]), name
    if tokens > 1:
        assert bits(out["q"]) != bits(_forward(c, heads, causal)["q"])


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", ROTARY, ids=map(_id, ROTARY))
def test_a_rotary_downstream_rotates_dq_and_dk_back_before_projects_products(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    c = _case(examples, tokens, features, heads, key_size)
    f = _rotary_forward(c, tokens, heads, causal)
    table = _table(tokens, key_size)
    result = attention_downstream_batch(
        to_array(c["delta"]),
        *(to_array(c[name]) for name in ("wq", "wk", "wv", "wo")),
        *(to_array(f[name]) for name in ("q", "k", "v", "p")),
        heads=heads,
        cos=to_array(table[0]),
        sin=to_array(table[1]),
    )
    out = {name: to_numpy(array) for name, array in zip(["dx", "dq", "dk", "dv"], result)}
    # the unrotated op's attend gradients from the rotated caches, then rotated back, then dx
    plain = _backward(c, f, heads)
    dq, dk = _rotated(plain["dq"], table, heads, inverse=True), _rotated(plain["dk"], table, heads, inverse=True)
    dx = (
        layer_downstream_batch(to_array(c["wq"]), to_array(dq))
        + layer_downstream_batch(to_array(c["wk"]), to_array(dk))
    ) + layer_downstream_batch(to_array(c["wv"]), to_array(plain["dv"]))
    assert bits(out["dq"]) == bits(dq)
    assert bits(out["dk"]) == bits(dk)
    assert bits(out["dv"]) == bits(plain["dv"])
    assert bits(out["dx"]) == bits(to_numpy(dx).reshape(c["x"].shape))


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", ROTARY, ids=map(_id, ROTARY))
def test_one_examples_rotary_pass_is_its_batch_rows(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    c = _case(examples, tokens, features, heads, key_size)
    one, batch = _rotary_forward(c, tokens, heads, causal, one=True), _rotary_forward(c, tokens, heads, causal)
    assert bits(one["a"]) == bits(batch["a"][0])
    for name in ["q", "k", "v", "p", "h"]:
        assert bits(one[name]) == bits(batch[name][:tokens]), name


def test_position_0_is_unrotated_so_one_token_is_the_unrotated_pass_by_bits():
    c = _case(2, 1, 6, 3, 2)
    f, plain = _rotary_forward(c, 1, 3, False), _forward(c, 3)
    for name in ["a", "q", "k", "v", "p", "h"]:
        assert bits(f[name]) == bits(plain[name]), name


def test_the_rotary_ops_refuse_half_a_table_an_odd_head_and_a_misshapen_table():
    c = _case(2, 3, 6, 3, 2)
    x, parameters = to_array(c["x"]), _parameters(c)
    cos, sin = (to_array(t) for t in _table(3, 2))
    with pytest.raises(ValueError, match="attention_forward_batch requires both cos and sin to rotate, or neither"):
        attention_forward_batch(x, *parameters, heads=3, cos=cos)
    with pytest.raises(ValueError, match=r"attention_forward_batch requires cos of shape Matrix\(3, 1\)"):
        attention_forward_batch(x, *parameters, heads=3, cos=Array([[1.0, 1.0]] * 3), sin=sin)
    with pytest.raises(ValueError, match=r"attention_forward_batch requires sin of shape Matrix\(3, 1\)"):
        attention_forward_batch(x, *parameters, heads=3, cos=cos, sin=Array([[0.0]] * 2))
    odd = _case(2, 3, 6, 2, 3)
    with pytest.raises(ValueError, match="attention_forward_batch requires an even head size to rotate, got d_k=3"):
        attention_forward_batch(to_array(odd["x"]), *_parameters(odd), heads=2, cos=cos, sin=sin)
    f = _forward(c, 3)
    args = (
        to_array(c["delta"]),
        *(to_array(c[name]) for name in ("wq", "wk", "wv", "wo")),
        *(to_array(f[name]) for name in ("q", "k", "v", "p")),
    )
    with pytest.raises(ValueError, match="attention_downstream_batch requires both cos and sin to rotate, or neither"):
        attention_downstream_batch(*args, heads=3, sin=sin)
    with pytest.raises(ValueError, match=r"attention_downstream_batch requires cos of shape Matrix\(3, 1\)"):
        attention_downstream_batch(*args, heads=3, cos=Array([[1.0]] * 2), sin=sin)
