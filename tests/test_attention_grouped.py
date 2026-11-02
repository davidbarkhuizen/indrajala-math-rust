"""
Grouped-query attention in the attention ops (indrajala-ml's grouped-query attention workplan, D2-D4):
`key_value_heads` `g` dividing `heads`, `wk`, `bk`, `wv`, `bv` and the caches `k`, `v` at `g * d_k`,
query head `i` reading key/value head `i // (h / g)`. Each op is checked by bits against
test_attention_ops' tiled composition at `g` (a group's `dk` and `dv` per key block its first head's
assigned and each later head's added, `dk` divided by the scale after), including dropout (the
masks per query head, as before), rotary positions (`K` rotated by its `g` heads) and the decode
step; `key_value_heads = heads` is the default by bits; and the ops refuse a `g` that doesn't divide
the heads and projections at the wrong width.
"""

from typing import Any

import numpy as np
import pytest
from array_helpers import bits, to_array, to_numpy
from test_attention_ops import (
    BLOCK,
    DROPOUT,
    FORWARD,
    _backward,
    _block_masks,
    _case,
    _composed_backward,
    _composed_forward,
    _forward,
    _gradients,
    _parameters,
    _projected,
    _rotated,
    _table,
)

from indrajala_math_rust import Array, Generator, attention_forward, attention_step, layer_accumulate_gradient_batch

# (examples, tokens, features, heads, key_size, key_value_heads): multi-query and groups of 2 and
# 3, one key block and several, widths off the lanes
GROUPED = [
    (1, 4, 6, 2, 3, 1),
    (2, 5, 6, 4, 2, 2),
    (2, 5, 6, 4, 2, 1),
    (3, 5, 6, 6, 2, 2),
    (2, 16, 32, 4, 8, 2),
    (2, 37, 8, 4, 4, 2),
    (1, 64, 64, 8, 8, 2),
    (3, 33, 6, 3, 2, 1),
]


def _id(shape: tuple[int, ...]) -> str:
    return "x".join(map(str, shape))


CASES = [(shape, block) for shape in GROUPED for block in (BLOCK, 3)]


def _case_id(case: tuple[tuple[int, ...], int]) -> str:
    return f"{_id(case[0])}-b{case[1]}"


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("case", CASES, ids=map(_case_id, CASES))
def test_a_grouped_forward_pass_is_the_tiled_composition(case: tuple[tuple[int, ...], int], causal: bool):
    (examples, tokens, features, heads, key_size, g), block = case
    c = _case(examples, tokens, features, heads, key_size, g)
    out = _forward(c, heads, causal, block=block, key_value_heads=g)
    composed = _composed_forward(c, tokens, heads, causal, block, key_value_heads=g)
    assert out["k"].shape == out["v"].shape == (examples * tokens, g * key_size)
    for name in ["a", "q", "k", "v", "h", "m", "l"]:
        assert out[name].shape == composed[name].shape, name
        assert bits(out[name]) == bits(composed[name]), name


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("case", CASES, ids=map(_case_id, CASES))
def test_a_grouped_downstream_is_the_tiled_composition(case: tuple[tuple[int, ...], int], causal: bool):
    (examples, tokens, features, heads, key_size, g), block = case
    c = _case(examples, tokens, features, heads, key_size, g)
    f = _forward(c, heads, causal, block=block, key_value_heads=g)
    out = _backward(c, f, heads, causal, block=block, key_value_heads=g)
    composed = _composed_backward(c, f, tokens, heads, causal, block, key_value_heads=g)
    assert out["dk"].shape == out["dv"].shape == (examples * tokens, g * key_size)
    for name in ["dx", "dq", "dk", "dv"]:
        assert out[name].shape == composed[name].shape, name
        assert bits(out[name]) == bits(composed[name]), name


@pytest.mark.parametrize("shape", GROUPED, ids=map(_id, GROUPED))
def test_the_grouped_gradients_are_each_projections_dense_gradients(shape: tuple[int, ...]):
    examples, tokens, features, heads, key_size, g = shape
    c = _case(examples, tokens, features, heads, key_size, g)
    f = _forward(c, heads, key_value_heads=g)
    b = _backward(c, f, heads, key_value_heads=g)
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


@pytest.mark.parametrize("shape", GROUPED, ids=map(_id, GROUPED))
def test_one_examples_grouped_pass_is_its_batch_rows(shape: tuple[int, ...]):
    examples, tokens, features, heads, key_size, g = shape
    c = _case(examples, tokens, features, heads, key_size, g)
    batch = _forward(c, heads, True, key_value_heads=g)
    for i in range(examples):
        result = attention_forward(to_array(c["x"][i]), *_parameters(c), heads=heads, causal=True, key_value_heads=g)
        single = {name: to_numpy(array) for name, array in zip(FORWARD[:7], result[:7]) if isinstance(array, Array)}
        assert bits(single["a"]) == bits(batch["a"][i])
        for name in ["q", "k", "v", "h", "m", "l"]:
            assert bits(single[name]) == bits(batch[name][i * tokens : (i + 1) * tokens]), name


@pytest.mark.parametrize("causal", [False, True])
def test_key_value_heads_equal_to_the_heads_is_the_default_by_bits(causal: bool):
    c = _case(2, 37, 8, 4, 4)
    default, explicit = _forward(c, 4, causal), _forward(c, 4, causal, key_value_heads=4)
    for name in ["a", "q", "k", "v", "h", "m", "l"]:
        assert bits(default[name]) == bits(explicit[name]), name
    b_default, b_explicit = _backward(c, default, 4, causal), _backward(c, default, 4, causal, key_value_heads=4)
    for name in ["dx", "dq", "dk", "dv"]:
        assert bits(b_default[name]) == bits(b_explicit[name]), name


def test_heads_sharing_a_key_value_head_with_the_same_queries_have_the_same_outputs():
    # every query head's block of wq and bq head 0's: within a group they read the same keys, so
    # their m, l and H are the same bits
    examples, tokens, features, heads, key_size, g = 2, 16, 32, 4, 8, 2
    c = _case(examples, tokens, features, heads, key_size, g)
    for name in ("wq", "bq"):
        c[name] = np.concatenate([c[name][:key_size]] * heads)
    out = _forward(c, heads, key_value_heads=g)
    for i in (1, 3):  # each group's second head against its first
        assert bits(out["h"][:, i * key_size : (i + 1) * key_size]) == bits(
            out["h"][:, (i - 1) * key_size : i * key_size]
        )
    assert bits(out["h"][:, 2 * key_size : 3 * key_size]) != bits(out["h"][:, :key_size])


# dropout: the masks per query head, numpy's (N, h, T, b) draws, as without groups


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", GROUPED, ids=map(_id, GROUPED))
def test_a_grouped_dropping_pass_is_the_composition_with_numpys_key_block_draws(shape: tuple[int, ...], causal: bool):
    examples, tokens, features, heads, key_size, g = shape
    c = _case(examples, tokens, features, heads, key_size, g)
    rng = Generator(7)
    out = _forward(c, heads, causal, dropout=DROPOUT, training=True, rng=rng, key_value_heads=g)
    masks = _block_masks(examples, heads, tokens, 7)
    keep = 1.0 - DROPOUT
    composed = _composed_forward(c, tokens, heads, causal, masks=masks, keep=keep, key_value_heads=g)
    for name in ["a", "h", "m", "l"]:
        assert bits(out[name]) == bits(composed[name]), name
    after = np.random.default_rng(7)
    after.random(examples * heads * tokens * tokens)
    assert rng.state == after.bit_generator.state
    b = _backward(c, out, heads, causal, dropout=DROPOUT, dropout_rng=out["rng"], key_value_heads=g)
    composed_b = _composed_backward(c, out, tokens, heads, causal, masks=masks, keep=keep, key_value_heads=g)
    for name in ["dx", "dq", "dk", "dv"]:
        assert bits(b[name]) == bits(composed_b[name]), name


# rotary positions: Q rotated by its h heads, K by its g


ROTARY = [shape for shape in GROUPED if shape[4] % 2 == 0]


def _rotary(c: dict[str, np.ndarray], tokens: int, heads: int, g: int, causal: bool) -> tuple[dict[str, Any], Any]:
    table = _table(tokens, c["bq"].shape[0] // heads)
    cos, sin = to_array(table[0]), to_array(table[1])
    return _forward(c, heads, causal, cos=cos, sin=sin, key_value_heads=g), table


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", ROTARY, ids=map(_id, ROTARY))
def test_a_grouped_rotary_pass_rotates_k_by_its_key_value_heads(shape: tuple[int, ...], causal: bool):
    examples, tokens, features, heads, key_size, g = shape
    c = _case(examples, tokens, features, heads, key_size, g)
    out, table = _rotary(c, tokens, heads, g, causal)
    q, k, v = _projected(c)
    rotated = (_rotated(q, table, heads), _rotated(k, table, g), v)
    composed = _composed_forward(c, tokens, heads, causal, qkv=rotated, key_value_heads=g)
    for name in ["a", "q", "k", "v", "h", "m", "l"]:
        assert bits(out[name]) == bits(composed[name]), name
    b = _backward(c, out, heads, causal, cos=to_array(table[0]), sin=to_array(table[1]), key_value_heads=g)
    plain = _backward(c, out, heads, causal, key_value_heads=g)
    assert bits(b["dq"]) == bits(_rotated(plain["dq"], table, heads, inverse=True))
    assert bits(b["dk"]) == bits(_rotated(plain["dk"], table, g, inverse=True))
    assert bits(b["dv"]) == bits(plain["dv"])


# the decode step: the caches at g * d_k


@pytest.mark.parametrize("rotary", [False, True])
@pytest.mark.parametrize("shape", ROTARY, ids=map(_id, ROTARY))
def test_each_grouped_step_is_the_causal_forward_pass_at_its_token_by_bits(shape: tuple[int, ...], rotary: bool):
    _examples, tokens, features, heads, key_size, g = shape
    c = _case(1, tokens, features, heads, key_size, g)
    c["x"] = c["x"][0]
    table = _table(tokens, key_size)
    rotation = {"cos": to_array(table[0]), "sin": to_array(table[1])} if rotary else {}
    result = attention_forward(
        to_array(c["x"]), *_parameters(c), heads=heads, causal=True, key_value_heads=g, **rotation
    )
    a, k, v = (to_numpy(r) for r in result[:1] + result[2:4])
    kv_width = g * key_size
    for t in range(tokens):
        caches = (to_array(k[:t]), to_array(v[:t])) if t else (Array.zeros((0, kv_width)),) * 2
        row = {"cos": to_array(table[0][t : t + 1]), "sin": to_array(table[1][t : t + 1])} if rotary else {}
        out, k_cache, v_cache = attention_step(
            to_array(c["x"][t * features : (t + 1) * features]),
            *_parameters(c),
            k_cache=caches[0],
            v_cache=caches[1],
            heads=heads,
            window=tokens,
            key_value_heads=g,
            **row,
        )
        assert bits(to_numpy(out)) == bits(a[t * features : (t + 1) * features]), t
        assert bits(to_numpy(k_cache).reshape(-1, kv_width)) == bits(k[: t + 1]), t
        assert bits(to_numpy(v_cache).reshape(-1, kv_width)) == bits(v[: t + 1]), t


# refusals


def test_the_ops_refuse_key_value_heads_that_dont_divide_the_heads_and_projections_at_the_wrong_width():
    c = _case(1, 3, 8, 4, 2, 2)
    parameters = _parameters(c)
    x = to_array(c["x"][0])
    for g in (0, 3, 8):
        with pytest.raises(ValueError, match=rf"key_value_heads >= 1 dividing heads=4, got key_value_heads={g}"):
            attention_forward(x, *parameters, heads=4, key_value_heads=g)
    # the default (every head its own) reads wk at h * d_k
    with pytest.raises(ValueError, match=r"attention_forward requires wk of shape Matrix\(8, 8\)"):
        attention_forward(x, *parameters, heads=4)
    f = _forward(c, 4, key_value_heads=2)
    with pytest.raises(ValueError, match=r"attention_downstream_batch requires k of shape Matrix\(3, 4\)"):
        _backward(c, {**f, "k": np.zeros((3, 8))}, 4, key_value_heads=2)
    cache = Array.zeros((2, 8))
    with pytest.raises(ValueError, match=r"attention_step requires k_cache of shape \(c, 4\)"):
        attention_step(
            to_array(c["x"][0][:8]), *parameters, k_cache=cache, v_cache=cache, heads=4, window=3, key_value_heads=2
        )
