"""
The cross-attention ops (`cross_attention_*`, indrajala-ml's cross-attention workplan, D3, D11, D13):
queries from the decoder's `T` tokens of `d` features, keys and values from the encoder's output,
the memory's `S` tokens of `d_m`, attend and combine as self-attention's. Each op's outputs are
checked by bits against the crate's own tiled composition (test_attention_ops.py's) over `T`
queries and `S` keys, project's products a dense layer's on the tokens' rows (`Q`) and the
memory's (`K`, `V`).

The mask is by segment alone, with no own-column exception: a query reads only its own segment's
keys, never a pad's, so a pad's query reads none, its `h` 0 and its output `bo` (the empty-row
guard: `l = 0` divided by 1 forward, and `m` read as 0, `l` as 1, backward). The backward gives
`dx = dQ Wq` and the memory's gradient `dm = dK Wk + dV Wv`. Dropout draws are self-attention's:
per key block of the memory, every example's, head's, query's and key's, numpy's `(N, h, T, b)`.
"""

import math
from typing import Any

import numpy as np
import pytest
from array_helpers import bits, to_array, to_numpy
from test_attention_ops import DROPOUT, Masking, key_blocks, tiled_backward, tiled_forward

import indrajala_math_rust as pa
from indrajala_math_rust import (
    Array,
    Generator,
    affine_forward_batch,
    attention_downstream_batch,
    attention_forward_batch,
    cross_attention_accumulate_gradient_batch,
    cross_attention_attend_batch,
    cross_attention_downstream_batch,
    cross_attention_forward_batch,
    layer_accumulate_gradient_batch,
    layer_downstream_batch,
)

NAMES = ["wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo"]
FORWARD = ["a", "q", "k", "v", "h", "m", "l", "rng"]
BACKWARD = ["dx", "dm", "dq", "dk", "dv"]

# (examples, tokens T, features d, memory tokens S, memory features d_m, heads, key_size,
# key_value_heads): one key and one query; S above and below T; d_m above, below and at d; one key
# block and several, off the 16; grouped key/value heads; the study's heads of 16 and 32
SHAPES = [
    (1, 1, 1, 1, 1, 1, 1, 1),
    (2, 3, 4, 1, 5, 2, 2, 2),
    (2, 5, 6, 9, 4, 3, 2, 3),
    (3, 9, 6, 4, 6, 2, 3, 1),
    (2, 17, 8, 37, 12, 4, 2, 2),
    (2, 40, 8, 23, 8, 2, 4, 1),
    (1, 33, 64, 48, 32, 4, 16, 2),
    (2, 20, 32, 20, 64, 1, 32, 1),
]


def _id(shape: tuple[int, ...]) -> str:
    return "x".join(map(str, shape))


def _case(shape: tuple[int, ...]) -> dict[str, Any]:
    # the crate's own PCG64, as test_attention_ops.py's: the same inputs on every platform
    examples, tokens, d, memory_tokens, d_m, heads, key_size, key_value_heads = shape
    rng = pa.default_rng(list(shape))

    def uniform(low: float, high: float, shape: Any) -> np.ndarray:
        return to_numpy(rng.uniform(low, high, shape))

    width, kv_width = heads * key_size, key_value_heads * key_size
    c: dict[str, Any] = {
        "x": uniform(-2.0, 2.0, (examples, tokens * d)),
        "memory": uniform(-2.0, 2.0, (examples, memory_tokens * d_m)),
    }
    fans = {"q": (width, d), "k": (kv_width, d_m), "v": (kv_width, d_m), "o": (d, width)}
    for n, (rows, fan_in) in fans.items():
        limit = 1.0 / math.sqrt(fan_in)
        c["w" + n] = uniform(-limit, limit, (rows, fan_in))
        c["b" + n] = uniform(-limit, limit, rows)
    c["delta"] = uniform(-1.0, 1.0, (examples, tokens * d))
    for name in NAMES:
        size = c[name].shape
        c["grad_" + name] = uniform(-1.0, 1.0, size if len(size) == 2 else size[0])
    c["shape"] = shape
    return c


def _parameters(c: dict[str, Any]) -> list[Array]:
    return [to_array(c[name]) for name in NAMES]


def _segments(examples: int, tokens: int, memory_tokens: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # D13's packed pairs: pads on the left, then pair k as segment k on both sides, each side's
    # lengths its own; a side may run out first, so a query's segment may have no keys
    rng = np.random.default_rng(seed)
    queries, keys = [], []
    for n in range(examples):
        pairs = int(rng.integers(1, 5))
        sides = []
        for length in (tokens, memory_tokens):
            pads = int(rng.integers(0, max(1, length // 3))) if n % 2 else 0
            row = [0.0] * pads
            for k in range(1, pairs + 1):
                row += [float(k)] * int(rng.integers(1, 13))
            sides.append((row + [0.0] * length)[:length] if len(row) < length else row[-length:])
        queries.append(sides[0])
        keys.append(sides[1])
    return np.array(queries), np.array(keys)


def _masking(segments: tuple[np.ndarray, np.ndarray] | None) -> Any:
    if segments is None:
        return lambda n: Masking(False)
    queries, keys = segments
    return lambda n: Masking(False, 0, (queries[n], keys[n]), own_column=False)


def _segment_options(segments: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Array]:
    if segments is None:
        return {}
    return {"segments": to_array(segments[0]), "memory_segments": to_array(segments[1])}


def _forward(c: dict[str, Any], segments: Any = None, **options: Any) -> dict[str, Any]:
    heads, key_value_heads = c["shape"][5], c["shape"][7]
    result = cross_attention_forward_batch(
        to_array(c["x"]),
        to_array(c["memory"]),
        *_parameters(c),
        heads=heads,
        key_value_heads=key_value_heads,
        **_segment_options(segments),
        **options,
    )
    return {name: (to_numpy(r) if isinstance(r, Array) else r) for name, r in zip(FORWARD, result)}


def _backward(c: dict[str, Any], f: dict[str, Any], segments: Any = None, **options: Any) -> dict[str, np.ndarray]:
    heads, key_value_heads = c["shape"][5], c["shape"][7]
    result = cross_attention_downstream_batch(
        to_array(c["delta"]),
        *(to_array(c[name]) for name in ("wq", "wk", "wv", "wo")),
        *(to_array(f[name]) for name in ("q", "k", "v", "h", "m", "l")),
        heads=heads,
        key_value_heads=key_value_heads,
        **_segment_options(segments),
        **options,
    )
    return {name: to_numpy(array) for name, array in zip(BACKWARD, result)}


def _rows(values: np.ndarray, width: int) -> Array:
    return to_array(values.reshape(-1, width))


def _composed_forward(c: dict[str, Any], segments: Any = None, block: int = 16, **kwargs: Any) -> dict[str, Any]:
    examples, tokens, d, memory_tokens, d_m, heads, _, key_value_heads = c["shape"]
    x_rows, memory_rows = _rows(c["x"], d), _rows(c["memory"], d_m)
    q = to_numpy(affine_forward_batch(to_array(c["wq"]), x_rows, to_array(c["bq"])))
    k = to_numpy(affine_forward_batch(to_array(c["wk"]), memory_rows, to_array(c["bk"])))
    v = to_numpy(affine_forward_batch(to_array(c["wv"]), memory_rows, to_array(c["bv"])))
    h, m, l = tiled_forward(
        (q, k, v),
        (examples, tokens, memory_tokens),
        heads,
        _masking(segments),
        block,
        key_value_heads=key_value_heads,
        **kwargs,
    )
    a = to_numpy(affine_forward_batch(to_array(c["wo"]), to_array(h), to_array(c["bo"])))
    return {"a": a.reshape(c["x"].shape), "q": q, "k": k, "v": v, "h": h, "m": m, "l": l}


def _composed_backward(
    c: dict[str, Any], f: dict[str, Any], segments: Any = None, block: int = 16, **kwargs: Any
) -> dict[str, np.ndarray]:
    examples, tokens, d, memory_tokens, _, heads, _, key_value_heads = c["shape"]
    dh = to_numpy(layer_downstream_batch(to_array(c["wo"]), _rows(c["delta"], d)))
    saved = tuple(f[name] for name in ("q", "k", "v", "h", "m", "l"))
    dq, dk, dv = tiled_backward(
        dh,
        saved,
        (examples, tokens, memory_tokens),
        heads,
        _masking(segments),
        block,
        key_value_heads=key_value_heads,
        **kwargs,
    )
    dx = to_numpy(layer_downstream_batch(to_array(c["wq"]), to_array(dq))).reshape(c["x"].shape)
    dm = layer_downstream_batch(to_array(c["wk"]), to_array(dk)) + layer_downstream_batch(
        to_array(c["wv"]), to_array(dv)
    )
    return {"dx": dx, "dm": to_numpy(dm), "dq": dq, "dk": dk, "dv": dv}


CASES = [(shape, block, segmented) for shape in SHAPES for block in (16, 3) for segmented in (False, True)]


def _case_id(case: tuple[tuple[int, ...], int, bool]) -> str:
    return f"{_id(case[0])}-b{case[1]}" + ("-segments" if case[2] else "")


def _case_segments(shape: tuple[int, ...], segmented: bool, seed: int = 0) -> Any:
    return _segments(shape[0], shape[1], shape[3], sum(shape) + seed) if segmented else None


@pytest.mark.parametrize("case", CASES, ids=map(_case_id, CASES))
def test_the_forward_pass_is_the_crates_tiled_composition(case: tuple[tuple[int, ...], int, bool]):
    shape, block, segmented = case
    c, segments = _case(shape), _case_segments(shape, segmented)
    out = _forward(c, segments, block=block)
    composed = _composed_forward(c, segments, block)
    assert out["rng"] is None
    for name in ["a", "q", "k", "v", "h", "m", "l"]:
        assert out[name].shape == composed[name].shape, name
        assert bits(out[name]) == bits(composed[name]), name


@pytest.mark.parametrize("case", CASES, ids=map(_case_id, CASES))
def test_the_downstream_is_the_crates_tiled_composition(case: tuple[tuple[int, ...], int, bool]):
    shape, block, segmented = case
    c, segments = _case(shape), _case_segments(shape, segmented)
    f = _forward(c, segments, block=block)
    out, composed = _backward(c, f, segments, block=block), _composed_backward(c, f, segments, block)
    examples, _, _, memory_tokens, d_m = shape[:5]
    assert out["dm"].shape == (examples * memory_tokens, d_m)
    for name in BACKWARD:
        assert out[name].shape == composed[name].shape, name
        assert bits(out[name]) == bits(composed[name]), name


@pytest.mark.parametrize("shape", SHAPES, ids=map(_id, SHAPES))
def test_skipping_blocks_no_query_reaches_gives_the_bits_of_computing_them(shape: tuple[int, ...]):
    # every row through every block: the wholly masked rows' e and P are 0, their exp(m - m') 1 or 0
    # (an empty row's), and the crate's sums are sequential folds, so the skipped zeros change nothing
    c, segments = _case(shape), _case_segments(shape, True, seed=1)
    f = _forward(c, segments, block=4)
    every = _composed_forward(c, segments, 4, skip=False)
    for name in ["h", "m", "l"]:
        assert bits(f[name]) == bits(every[name]), name
    out, composed = _backward(c, f, segments, block=4), _composed_backward(c, f, segments, 4, skip=False)
    for name in BACKWARD:
        assert bits(out[name]) == bits(composed[name]), name


@pytest.mark.parametrize("shape", SHAPES, ids=map(_id, SHAPES))
def test_the_gradients_are_each_projections_dense_gradients_q_on_the_tokens_k_and_v_on_the_memory(
    shape: tuple[int, ...],
):
    c, segments = _case(shape), _case_segments(shape, True)
    d, d_m = shape[2], shape[4]
    f = _forward(c, segments)
    b = _backward(c, f, segments)
    grads = cross_attention_accumulate_gradient_batch(
        to_array(c["delta"]),
        to_array(c["x"]),
        to_array(c["memory"]),
        to_array(f["h"]),
        *(to_array(b[name]) for name in ("dq", "dk", "dv")),
        *(to_array(c["grad_" + name]) for name in NAMES),
    )
    x_rows, memory_rows = _rows(c["x"], d), _rows(c["memory"], d_m)
    inputs = {"q": (b["dq"], x_rows), "k": (b["dk"], memory_rows), "v": (b["dv"], memory_rows)}
    inputs["o"] = (c["delta"].reshape(-1, d), to_array(f["h"]))
    for i, n in enumerate("qkvo"):
        d_rows, rows = inputs[n]
        expected = layer_accumulate_gradient_batch(
            to_array(d_rows), rows, to_array(c["grad_w" + n]), to_array(c["grad_b" + n])
        )
        assert bits(to_numpy(grads[2 * i])) == bits(to_numpy(expected[0])), "w" + n
        assert bits(to_numpy(grads[2 * i + 1])) == bits(to_numpy(expected[1])), "b" + n


def test_a_pad_query_reads_no_key_its_output_the_bias_and_its_gradients_zero():
    shape = (2, 6, 4, 5, 3, 2, 2, 1)
    c = _case(shape)
    queries = np.array([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    keys = np.array([[0.0, 1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    f = _forward(c, (queries, keys), block=2)
    pads = np.flatnonzero(queries.reshape(-1) == 0.0)
    assert np.all(f["h"][pads] == 0.0) and np.all(f["l"][pads] == 0.0) and np.all(f["m"][pads] == -np.inf)
    a = f["a"].reshape(-1, shape[2])
    assert all(bits(a[r]) == bits(c["bo"]) for r in pads)
    b = _backward(c, f, (queries, keys), block=2)
    assert np.all(b["dq"][pads] == 0.0) and np.all(np.isfinite(b["dm"]))
    # a pad key's gradients are 0: no query reads it
    assert np.all(b["dk"][[0]] == 0.0) and np.all(b["dv"][[0]] == 0.0)


def test_a_pad_keys_memory_and_another_segments_change_nothing():
    shape = (1, 6, 4, 7, 3, 2, 2, 2)
    c = _case(shape)
    queries = np.array([[0.0, 1.0, 1.0, 1.0, 2.0, 2.0]])
    keys = np.array([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0]])
    before = _forward(c, (queries, keys), block=3)
    moved = dict(c, memory=c["memory"].copy())
    # the pads' memory (keys 0 and 1) and segment 2's (keys 4..6), for segment 1's queries (1..3)
    d_m = shape[4]
    moved["memory"][0, : 2 * d_m] += 5.0
    after = _forward(moved, (queries, keys), block=3)
    assert bits(after["a"]) == bits(before["a"])
    moved["memory"][0, 4 * d_m :] -= 3.0
    later = _forward(moved, (queries, keys), block=3)
    rows = slice(1 * shape[2], 4 * shape[2])
    assert bits(later["a"][0, rows]) == bits(before["a"][0, rows])
    assert bits(later["a"][0, 4 * shape[2] :]) != bits(before["a"][0, 4 * shape[2] :])


@pytest.mark.parametrize("shape", [s for s in SHAPES if s[3] == 1], ids=_id)
def test_with_one_memory_token_every_weight_is_one_so_h_is_its_value(shape: tuple[int, ...]):
    c = _case(shape)
    out = _forward(c)
    assert np.all(out["l"] == 1.0)
    examples, tokens, heads, key_size, key_value_heads = shape[0], shape[1], shape[5], shape[6], shape[7]
    group = heads // key_value_heads
    for r in range(examples * tokens):
        for i in range(heads):
            j = i // group
            assert bits(out["h"][r, i * key_size : (i + 1) * key_size]) == bits(
                out["v"][r // tokens, j * key_size : (j + 1) * key_size]
            )


@pytest.mark.parametrize("shape", [s for s in SHAPES if s[2] == s[4]], ids=_id)
def test_with_the_tokens_as_the_memory_and_no_segments_it_is_unmasked_self_attention_by_bits(shape: tuple[int, ...]):
    # S = T and d_m = d: the same products on the same rows, the same tiles and draws
    examples, tokens, d, _, _, heads, key_size, key_value_heads = shape
    c = _case((examples, tokens, d, tokens, d, heads, key_size, key_value_heads))
    c["memory"] = c["x"]
    rng_self, rng_cross = Generator(5), Generator(5)
    plain = attention_forward_batch(
        to_array(c["x"]),
        *_parameters(c),
        heads=heads,
        key_value_heads=key_value_heads,
        dropout=DROPOUT,
        training=True,
        rng=rng_self,
        block=5,
    )
    cross = _forward(c, dropout=DROPOUT, training=True, rng=rng_cross, block=5)
    for name, array in zip(FORWARD[:7], plain[:7]):
        assert bits(to_numpy(array)) == bits(cross[name]), name
    assert rng_self.state == rng_cross.state
    self_b = attention_downstream_batch(
        to_array(c["delta"]),
        *[to_array(c[name]) for name in ("wq", "wk", "wv", "wo")] + list(plain[1:7]),
        heads=heads,
        key_value_heads=key_value_heads,
        dropout=DROPOUT,
        dropout_rng=plain[7],
        block=5,
    )
    cross_b = _backward(c, cross, dropout=DROPOUT, dropout_rng=cross["rng"], block=5)
    for name, array in zip(["dq", "dk", "dv"], self_b[1:]):
        assert bits(to_numpy(array)) == bits(cross_b[name]), name


def _block_masks(examples: int, heads: int, tokens: int, memory_tokens: int, seed: int, block: int) -> list[np.ndarray]:
    # numpy's draws key block by key block of the memory, each (N, h, T, b)
    rng = np.random.default_rng(seed)
    return [
        (rng.random((examples, heads, tokens, k1 - k0)) >= DROPOUT).astype(np.float64)
        for k0, k1 in key_blocks(memory_tokens, block)
    ]


@pytest.mark.parametrize("shape", SHAPES, ids=map(_id, SHAPES))
def test_a_dropping_pass_is_the_composition_with_numpys_key_block_draws(shape: tuple[int, ...]):
    examples, tokens, _, memory_tokens, _, heads = shape[:6]
    c, segments = _case(shape), _case_segments(shape, True, seed=2)
    rng = Generator(7)
    out = _forward(c, segments, dropout=DROPOUT, training=True, rng=rng, block=3)
    masks = _block_masks(examples, heads, tokens, memory_tokens, 7, 3)
    keep = 1.0 - DROPOUT
    composed = _composed_forward(c, segments, 3, masks=masks, keep=keep)
    for name in ["a", "q", "k", "v", "h", "m", "l"]:
        assert bits(out[name]) == bits(composed[name]), name
    after = np.random.default_rng(7)
    after.random(examples * heads * tokens * memory_tokens)
    assert rng.state == after.bit_generator.state
    assert out["rng"].state == Generator(7).state
    b = _backward(c, out, segments, dropout=DROPOUT, dropout_rng=out["rng"], block=3)
    composed_b = _composed_backward(c, out, segments, 3, masks=masks, keep=keep)
    for name in BACKWARD:
        assert bits(b[name]) == bits(composed_b[name]), name
    assert out["rng"].state == Generator(7).state


def test_without_training_nothing_is_drawn_and_the_bits_are_as_without_dropout():
    shape = SHAPES[4]
    c, segments = _case(shape), _case_segments(shape, True)
    before = _forward(c, segments)
    rng = Generator(4)
    state = rng.state
    out = _forward(c, segments, dropout=DROPOUT, training=False, rng=rng)
    assert out["rng"] is None and rng.state == state
    for name in ["a", "h", "m", "l"]:
        assert bits(out[name]) == bits(before[name]), name


def test_the_cross_attention_ops_refuse_what_doesnt_fit():
    shape = (2, 3, 4, 5, 6, 2, 2, 1)
    c = _case(shape)
    x, memory = to_array(c["x"]), to_array(c["memory"])
    queries, _ = _segments(2, 3, 5, 0)
    with pytest.raises(ValueError, match="both segments and memory_segments, or neither"):
        cross_attention_forward_batch(
            x, memory, *_parameters(c), heads=2, key_value_heads=1, segments=to_array(queries)
        )
    with pytest.raises(ValueError, match="a segment id per token"):
        cross_attention_forward_batch(
            x,
            memory,
            *_parameters(c),
            heads=2,
            key_value_heads=1,
            segments=to_array(queries),
            memory_segments=to_array(queries),
        )
    with pytest.raises(ValueError, match="one memory row per example"):
        cross_attention_forward_batch(x, to_array(c["memory"][:1]), *_parameters(c), heads=2, key_value_heads=1)
    with pytest.raises(ValueError, match="whole tokens of 6 features"):
        cross_attention_forward_batch(x, to_array(c["memory"][:, :-1]), *_parameters(c), heads=2, key_value_heads=1)
    with pytest.raises(ValueError, match="requires a 2D memory"):
        cross_attention_forward_batch(x, to_array(c["memory"][0]), *_parameters(c), heads=2, key_value_heads=1)
    wrong = _parameters(c)
    wrong[4] = to_array(c["wv"][:, :-1])
    with pytest.raises(ValueError, match="requires wv of shape"):
        cross_attention_forward_batch(x, memory, *wrong, heads=2, key_value_heads=1)
    with pytest.raises(ValueError, match="requires an rng to drop out in training"):
        cross_attention_forward_batch(
            x, memory, *_parameters(c), heads=2, key_value_heads=1, dropout=0.1, training=True
        )
    f = _forward(c)
    with pytest.raises(ValueError, match="k of whole memories"):
        cross_attention_downstream_batch(
            to_array(c["delta"]),
            *[to_array(c[name]) for name in ("wq", "wk", "wv", "wo")]
            + [to_array(f["q"]), to_array(f["k"][:-1])]
            + [to_array(f[name]) for name in ("v", "h", "m", "l")],
            heads=2,
            key_value_heads=1,
        )


# decoding (indrajala-ml's cross-attention workplan, D7): the queries against the keys and values
# computed once per source, the forward pass's k and v, give its output by bits


def _attend(c: dict[str, Any], f: dict[str, Any], x: np.ndarray, segments: Any = None, **options: Any) -> np.ndarray:
    heads, key_value_heads = c["shape"][5], c["shape"][7]
    return to_numpy(
        cross_attention_attend_batch(
            to_array(x),
            to_array(f["k"]),
            to_array(f["v"]),
            *(to_array(c[name]) for name in ("wq", "bq", "wo", "bo")),
            heads=heads,
            key_value_heads=key_value_heads,
            **_segment_options(segments),
            **options,
        )
    )


@pytest.mark.parametrize("case", CASES, ids=map(_case_id, CASES))
def test_attending_from_the_cached_keys_and_values_is_the_forward_pass_by_bits(case: tuple[tuple[int, ...], int, bool]):
    shape, block, segmented = case
    c, segments = _case(shape), _case_segments(shape, segmented)
    f = _forward(c, segments, block=block)
    assert bits(_attend(c, f, c["x"], segments, block=block)) == bits(f["a"])


@pytest.mark.parametrize("shape", SHAPES, ids=map(_id, SHAPES))
def test_one_query_a_row_is_the_forward_passs_row_at_that_token_by_bits(shape: tuple[int, ...]):
    # a decode's step: each token alone against the whole memory, its own segment's mask
    _, tokens, d = shape[:3]
    c, segments = _case(shape), _case_segments(shape, True)
    f = _forward(c, segments)
    for t in range(tokens):
        x = c["x"][:, t * d : (t + 1) * d]
        one = (segments[0][:, t : t + 1], segments[1])
        assert bits(_attend(c, f, x, one)) == bits(f["a"][:, t * d : (t + 1) * d]), t


def test_the_attend_op_refuses_what_doesnt_fit():
    shape = (2, 3, 4, 5, 6, 2, 2, 1)
    c = _case(shape)
    f = _forward(c)
    with pytest.raises(ValueError, match="k of whole memories"):
        cross_attention_attend_batch(
            to_array(c["x"]),
            to_array(f["k"][:-1]),
            to_array(f["v"]),
            *(to_array(c[name]) for name in ("wq", "bq", "wo", "bo")),
            heads=2,
            key_value_heads=1,
        )
    with pytest.raises(ValueError, match="requires v of shape"):
        cross_attention_attend_batch(
            to_array(c["x"]),
            to_array(f["k"]),
            to_array(f["v"][:, :-1]),
            *(to_array(c[name]) for name in ("wq", "bq", "wo", "bo")),
            heads=2,
            key_value_heads=1,
        )
