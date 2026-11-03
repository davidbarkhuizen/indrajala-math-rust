"""
The attention ops (`attention_*`, indrajala-ml's multi-head attention workplan, D4; its context-length
workplan, D3, D12): one call per pass, project, attend and combine, attend tiled. Each op's outputs
are checked by bits against the crate's own unfused composition, built here from the crate's
existing ops: the projections, the downstream products and the gradients as a dense layer's
(`affine_forward_batch`, `layer_downstream_batch`, `layer_accumulate_gradient_batch`) on the
`(N * T, d)` and `(N * T, h * d_k)` rows; the products between activations as `@` per example, head
and tile; `exp` as the crate's `exp`; each row's sum a left fold. The elementwise steps between them
are numpy's, exact. numpy's own products (BLAS) are not the crate's, so numpy is the reference only
where no product is involved, and in the exact tests.

The tiled attend (D3): per key block of `block` (cut at the columns that are multiples of it), in
order, per example the rows reaching it, and per head and row `m' = max(m, the block's row max)`,
`e = exp(S - m')` (0 in place of an all-masked row's `m'`), `l = l exp(m - m') + fold(e)`, `H = H exp(m
- m') + e V`, then `H / l`; the backward recomputes `P = exp(S - m) / l`, takes `D = fold(dH H)` over
`d_k`, `dS = P (dP - D)`, `dQ` summed over the blocks in order. Masked scores are `-inf`: causal, and
by segment (test_attention_segments.py).

Dropout (D12): per key block, a mask `u >= dropout` for every example, head, query and key in that
order, numpy's `(N, h, T, b)` draw, so a pass draws `N h T T` values; `e` is dropped as `e M / keep`
before `e V`, and the backward draws the same masks again from the forward's copy of the generator:
`dV = (P kept)^T dH`, `dP = dP kept`, `kept = M / keep`.
"""

import math
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from array_helpers import bits, to_array, to_numpy

import indrajala_math_rust as pa
from indrajala_math_rust import (
    Array,
    Generator,
    affine_forward_batch,
    attention_accumulate_gradient_batch,
    attention_downstream_batch,
    attention_forward,
    attention_forward_batch,
    layer_accumulate_gradient_batch,
    layer_downstream_batch,
)

BLOCK = 16

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
# past one key block of 16: the study's 4 heads of 16, and widths off the lanes
LONG = [(2, 37, 8, 2, 4), (1, 64, 64, 4, 16), (3, 33, 6, 3, 2)]
SHAPES = ONE_HEAD + MULTI_HEAD + LONG
NAMES = ["wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo"]


def _id(shape: tuple[int, ...]) -> str:
    return "x".join(map(str, shape))


def _case(
    examples: int, tokens: int, features: int, heads: int, key_size: int, key_value_heads: int | None = None
) -> dict[str, np.ndarray]:
    # the crate's own PCG64, numpy's `default_rng` stream with `uniform` unfused: the same inputs on
    # every platform, where numpy's own `uniform` contracts to an FMA on some (macOS and Windows on
    # arm64); wk, bk, wv and bv at key_value_heads heads when given (grouped-query attention)
    seed = [examples, tokens, features, heads, key_size]
    rng = pa.default_rng(seed if key_value_heads is None else [*seed, key_value_heads])

    def uniform(low: float, high: float, shape: Any) -> np.ndarray:
        return to_numpy(rng.uniform(low, high, shape))

    d, width = features, heads * key_size
    kv_width = width if key_value_heads is None else key_value_heads * key_size
    c = {"x": uniform(-2.0, 2.0, (examples, tokens * d))}
    for name in NAMES:
        rows, fan_in = (d, width) if name[1] == "o" else (kv_width if name[1] in "kv" else width, d)
        limit = 1.0 / math.sqrt(fan_in)
        c[name] = uniform(-limit, limit, (rows, fan_in) if name[0] == "w" else rows)
    c["delta"] = uniform(-1.0, 1.0, (examples, tokens * d))
    for name in NAMES:
        shape = c[name].shape
        c["grad_" + name] = uniform(-1.0, 1.0, shape if len(shape) == 2 else shape[0])
    return c


def _parameters(c: dict[str, np.ndarray]) -> list[Array]:
    return [to_array(c[name]) for name in NAMES]


def _head(rows: np.ndarray, n: int, i: int, tokens: int, width: int) -> np.ndarray:
    # example n's head i: rows n * T.., columns i * width..
    return rows[n * tokens : (n + 1) * tokens, i * width : (i + 1) * width]


def _side_by_side(blocks: list[list[np.ndarray]]) -> np.ndarray:
    # blocks[n][i], each (T, width), as (N * T, h * width) rows
    return np.concatenate([np.concatenate(heads, axis=1) for heads in blocks])


# the tiled composition


def key_blocks(keys: int, block: int, origin: int = 0) -> list[tuple[int, int]]:
    # bounds at the columns that are multiples of block, key 0 at column origin
    first = (block - origin % block) % block or block
    bounds = [0, *range(first, keys, block), keys]
    return [(a, b) for a, b in pairwise(bounds) if a < b]


class Masking:
    """
    One example's: causal (key u at most query t's column t + offset) and by segment ids, the own
    column always allowed unless not own_column (cross-attention's two sequences).
    """

    def __init__(
        self,
        causal: bool,
        offset: int = 0,
        segments: tuple[np.ndarray, np.ndarray] | None = None,
        own_column: bool = True,
    ):
        self.causal, self.offset, self.segments, self.own_column = causal, offset, segments, own_column

    def allowed(self, t: int, u: int) -> bool:
        column = t + self.offset
        if self.causal and u > column:
            return False
        if self.segments is None:
            return True
        queries, keys = self.segments
        return (self.own_column and u == column) or (keys[u] == queries[t] and keys[u] != 0.0)

    def rows(self, queries: int, k0: int, k1: int, skip: bool = True) -> tuple[int, int] | None:
        if not skip:
            return 0, queries
        r0 = max(0, k0 - self.offset) if self.causal else 0
        if r0 >= queries:
            return None
        if self.segments is None:
            return r0, queries
        reached = [t for t in range(r0, queries) if any(self.allowed(t, u) for u in range(k0, k1))]
        return (reached[0], reached[-1] + 1) if reached else None


def _fold(values: Any) -> float:
    total = float(values[0])
    for x in values[1:]:
        total += float(x)
    return total


def _exp(values: np.ndarray) -> np.ndarray:
    return to_numpy(pa.exp(to_array(np.asarray(values, dtype=np.float64).reshape(-1)))).reshape(np.shape(values))


def _scores(q: np.ndarray, k: np.ndarray, masking: Masking, r0: int, k0: int, d_k: int) -> np.ndarray:
    s = to_numpy(to_array(q) @ to_array(k).T) / math.sqrt(d_k)
    for r in range(s.shape[0]):
        for j in range(s.shape[1]):
            if not masking.allowed(r0 + r, k0 + j):
                s[r, j] = -np.inf
    return s


def tiled_forward(
    qkv: tuple[np.ndarray, np.ndarray, np.ndarray],
    extent: tuple[int, int, int],
    heads: int,
    masking: Any,
    block: int = BLOCK,
    origin: int = 0,
    masks: list[np.ndarray] | None = None,
    keep: float = 1.0,
    skip: bool = True,
    key_value_heads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # (h, m, l) packed as the ops': masking(n) an example's Masking, masks[b] block b's (N, h, T_q, b);
    # query head i reads key/value head i // (h / g)
    q, k, v = qkv
    examples, queries, keys = extent
    width = q.shape[1]
    d_k = width // heads
    group = heads // (heads if key_value_heads is None else key_value_heads)
    m = np.full((examples * queries, heads), -np.inf)
    l = np.zeros((examples * queries, heads))
    h = np.zeros((examples * queries, width))
    for b, (k0, k1) in enumerate(key_blocks(keys, block, origin)):
        for n in range(examples):
            mask_n = masking(n)
            rows = mask_n.rows(queries, k0, k1, skip)
            if rows is None:
                continue
            r0, r1 = rows
            for i in range(heads):
                cols = slice(i * d_k, (i + 1) * d_k)
                kv_cols = slice(i // group * d_k, (i // group + 1) * d_k)
                s = _scores(
                    q[n * queries + r0 : n * queries + r1, cols],
                    k[n * keys + k0 : n * keys + k1, kv_cols],
                    mask_n,
                    r0,
                    k0,
                    d_k,
                )
                e = np.zeros_like(s)
                alphas = []
                for r in range(r1 - r0):
                    at = n * queries + r0 + r
                    m_old = m[at, i]
                    m_new = max(m_old, *s[r].tolist())
                    shift = 0.0 if m_new == -np.inf else m_new
                    e[r] = _exp(s[r] - shift)
                    alpha = float(_exp(np.array([m_old - shift]))[0])
                    l[at, i] = l[at, i] * alpha + _fold(e[r])
                    m[at, i] = m_new
                    alphas.append(alpha)
                    if masks is not None:
                        e[r] = e[r] * masks[b][n, i, r0 + r] / keep
                ev = to_numpy(to_array(e) @ to_array(v[n * keys + k0 : n * keys + k1, kv_cols]))
                for r in range(r1 - r0):
                    at = n * queries + r0 + r
                    h[at, cols] = h[at, cols] * alphas[r] + ev[r]
    # a row no key is allowed to (a cross-attention pad's query) keeps l = 0 and h = 0, divided by 1
    sums = np.where(l == 0.0, 1.0, l)
    for i in range(heads):
        h[:, i * d_k : (i + 1) * d_k] /= sums[:, i : i + 1]
    return h, m, l


def tiled_backward(
    dh: np.ndarray,
    saved: tuple[np.ndarray, ...],
    extent: tuple[int, int, int],
    heads: int,
    masking: Any,
    block: int = BLOCK,
    masks: list[np.ndarray] | None = None,
    keep: float = 1.0,
    skip: bool = True,
    key_value_heads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # a group's dk and dv per key block: its first head's assigned, each later head's added, in
    # head order; dk divided by the scale once, after
    q, k, v, h, m, l = saved
    # a row no key is allowed to (m = -inf, l = 0) reads m as 0 and l as 1: its p is 0
    m, l = np.where(m == -np.inf, 0.0, m), np.where(l == 0.0, 1.0, l)
    examples, queries, keys = extent
    width = q.shape[1]
    d_k = width // heads
    group = heads // (heads if key_value_heads is None else key_value_heads)
    s_k = math.sqrt(d_k)
    term = np.array(
        [
            [_fold(dh[r, i * d_k : (i + 1) * d_k] * h[r, i * d_k : (i + 1) * d_k]) for i in range(heads)]
            for r in range(dh.shape[0])
        ]
    )
    dq, dk, dv = np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    for b, (k0, k1) in enumerate(key_blocks(keys, block)):
        for n in range(examples):
            mask_n = masking(n)
            rows = mask_n.rows(queries, k0, k1, skip)
            if rows is None:
                continue
            r0, r1 = rows
            ra, ka = slice(n * queries + r0, n * queries + r1), slice(n * keys + k0, n * keys + k1)
            for i in range(heads):
                cols = slice(i * d_k, (i + 1) * d_k)
                kv_cols = slice(i // group * d_k, (i // group + 1) * d_k)
                s = _scores(q[ra, cols], k[ka, kv_cols], mask_n, r0, k0, d_k)
                p = _exp(s - m[ra, i : i + 1]) / l[ra, i : i + 1]
                dp = to_numpy(to_array(dh[ra, cols]) @ to_array(v[ka, kv_cols]).T)
                if masks is None:
                    dv_t = to_numpy(to_array(p).T @ to_array(dh[ra, cols]))
                else:
                    kept = masks[b][n, i, r0:r1] / keep
                    dv_t = to_numpy(to_array(p * kept).T @ to_array(dh[ra, cols]))
                    dp = dp * kept
                ds = p * (dp - term[ra, i : i + 1])
                dq[ra, cols] += to_numpy(to_array(ds) @ to_array(k[ka, kv_cols]))
                dk_t = to_numpy(to_array(ds).T @ to_array(q[ra, cols]))
                if i % group == 0:
                    dk[ka, kv_cols], dv[ka, kv_cols] = dk_t, dv_t
                else:
                    dk[ka, kv_cols] += dk_t
                    dv[ka, kv_cols] += dv_t
    return dq / s_k, dk / s_k, dv


def _masking(causal: bool, segments: np.ndarray | None = None) -> Any:
    return lambda n: Masking(causal, 0, None if segments is None else (segments[n], segments[n]))


FORWARD = ["a", "q", "k", "v", "h", "m", "l", "rng"]


def _forward(c: dict[str, np.ndarray], heads: int, causal: bool = False, **options: Any) -> dict[str, Any]:
    result = attention_forward_batch(to_array(c["x"]), *_parameters(c), heads=heads, causal=causal, **options)
    return {name: (to_numpy(r) if isinstance(r, Array) else r) for name, r in zip(FORWARD, result)}


def _projected(c: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    rows = to_array(c["x"].reshape(-1, c["bo"].shape[0]))
    return tuple(to_numpy(affine_forward_batch(to_array(c["w" + n]), rows, to_array(c["b" + n]))) for n in "qkv")


def _composed_forward(
    c: dict[str, np.ndarray], tokens: int, heads: int, causal: bool = False, block: int = BLOCK, **kwargs: Any
) -> dict[str, np.ndarray]:
    q, k, v = kwargs.pop("qkv", None) or _projected(c)
    examples = c["x"].shape[0]
    h, m, l = tiled_forward(
        (q, k, v), (examples, tokens, tokens), heads, kwargs.pop("masking", _masking(causal)), block, **kwargs
    )
    a = to_numpy(affine_forward_batch(to_array(c["wo"]), to_array(h), to_array(c["bo"])))
    return {"a": a.reshape(c["x"].shape), "q": q, "k": k, "v": v, "h": h, "m": m, "l": l}


def _backward(
    c: dict[str, np.ndarray], f: dict[str, Any], heads: int, causal: bool = False, **options: Any
) -> dict[str, np.ndarray]:
    result = attention_downstream_batch(
        to_array(c["delta"]),
        *(to_array(c[name]) for name in ("wq", "wk", "wv", "wo")),
        *(to_array(f[name]) for name in ("q", "k", "v", "h", "m", "l")),
        heads=heads,
        causal=causal,
        **options,
    )
    return {name: to_numpy(array) for name, array in zip(["dx", "dq", "dk", "dv"], result)}


def _composed_backward(
    c: dict[str, np.ndarray],
    f: dict[str, Any],
    tokens: int,
    heads: int,
    causal: bool = False,
    block: int = BLOCK,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    d = c["bo"].shape[0]
    dh = to_numpy(layer_downstream_batch(to_array(c["wo"]), to_array(c["delta"].reshape(-1, d))))
    saved = tuple(f[name] for name in ("q", "k", "v", "h", "m", "l"))
    examples = c["x"].shape[0]
    masking = kwargs.pop("masking", _masking(causal))
    dq, dk, dv = tiled_backward(dh, saved, (examples, tokens, tokens), heads, masking, block, **kwargs)
    dx = (
        layer_downstream_batch(to_array(c["wq"]), to_array(dq))
        + layer_downstream_batch(to_array(c["wk"]), to_array(dk))
    ) + layer_downstream_batch(to_array(c["wv"]), to_array(dv))
    return {"dx": to_numpy(dx).reshape(c["x"].shape), "dq": dq, "dk": dk, "dv": dv}


def _gradients(c: dict[str, np.ndarray], f: dict[str, Any], b: dict[str, np.ndarray]) -> list[np.ndarray]:
    grads = attention_accumulate_gradient_batch(
        to_array(c["delta"]),
        to_array(c["x"]),
        to_array(f["h"]),
        *(to_array(b[name]) for name in ("dq", "dk", "dv")),
        *(to_array(c["grad_" + name]) for name in NAMES),
    )
    return [to_numpy(g) for g in grads]


CASES = [(shape, block) for shape in SHAPES for block in (BLOCK, 3)]


def _case_id(case: tuple[tuple[int, ...], int]) -> str:
    return f"{_id(case[0])}-b{case[1]}"


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("case", CASES, ids=map(_case_id, CASES))
def test_the_forward_pass_is_the_crates_tiled_composition(case: tuple[tuple[int, ...], int], causal: bool):
    (examples, tokens, features, heads, key_size), block = case
    c = _case(examples, tokens, features, heads, key_size)
    out = _forward(c, heads, causal, block=block)
    composed = _composed_forward(c, tokens, heads, causal, block)
    assert out["rng"] is None
    for name in ["a", "q", "k", "v", "h", "m", "l"]:
        assert out[name].shape == composed[name].shape, name
        assert bits(out[name]) == bits(composed[name]), name


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("case", CASES, ids=map(_case_id, CASES))
def test_the_downstream_is_the_crates_tiled_composition(case: tuple[tuple[int, ...], int], causal: bool):
    (examples, tokens, features, heads, key_size), block = case
    c = _case(examples, tokens, features, heads, key_size)
    f = _forward(c, heads, causal, block=block)
    out, composed = _backward(c, f, heads, causal, block=block), _composed_backward(c, f, tokens, heads, causal, block)
    for name in ["dx", "dq", "dk", "dv"]:
        assert out[name].shape == composed[name].shape, name
        assert bits(out[name]) == bits(composed[name]), name


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", LONG, ids=map(_id, LONG))
def test_skipping_a_block_no_query_reaches_gives_the_bits_of_computing_it(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    # every row through every block: a wholly masked block's e is 0 and its exp(m - m') 1, and the
    # crate's sums are sequential folds, so the skipped zeros change nothing (D3)
    c = _case(examples, tokens, features, heads, key_size)
    f = _forward(c, heads, causal, block=5)
    every = _composed_forward(c, tokens, heads, causal, 5, skip=False)
    for name in ["h", "m", "l"]:
        assert bits(f[name]) == bits(every[name]), name
    out, composed = (
        _backward(c, f, heads, causal, block=5),
        _composed_backward(c, f, tokens, heads, causal, 5, skip=False),
    )
    for name in ["dx", "dq", "dk", "dv"]:
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
        single = {name: to_numpy(array) for name, array in zip(FORWARD[:7], result[:7]) if isinstance(array, Array)}
        assert bits(single["a"]) == bits(batch["a"][i])
        for name in ["q", "k", "v", "h", "m", "l"]:
            assert single[name].shape == (tokens, batch[name].shape[1]), name
            assert bits(single[name]) == bits(batch[name][i * tokens : (i + 1) * tokens]), name


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


@pytest.mark.parametrize("examples,features,heads,key_size", [(1, 1, 1, 1), (3, 5, 1, 5), (2, 32, 4, 8), (2, 6, 3, 5)])
def test_with_one_token_attention_is_the_value_then_the_output_projection(
    examples: int, features: int, heads: int, key_size: int
):
    # every head's one weight is exp(0) / 1 = 1, so h is v and out = (X Wv^T + bv) Wo^T + bo by bits
    c = _case(examples, 1, features, heads, key_size)
    out = _forward(c, heads)
    assert np.all(out["l"] == 1.0)
    assert bits(out["h"]) == bits(out["v"])
    v = affine_forward_batch(to_array(c["wv"]), to_array(c["x"]), to_array(c["bv"]))
    expected = affine_forward_batch(to_array(c["wo"]), v, to_array(c["bo"]))
    assert bits(out["a"]) == bits(to_numpy(expected))


@pytest.mark.parametrize("examples,features,heads,key_size", [(1, 1, 1, 1), (2, 7, 1, 7), (2, 32, 4, 8)])
def test_with_zero_query_and_key_weights_every_weight_is_one_so_l_counts_the_keys(
    examples: int, features: int, heads: int, key_size: int
):
    # every score 0: m = 0, every e = exp(0) = 1, l = 16 exactly, and H the keys' V summed, / 16
    tokens = 16
    c = _case(examples, tokens, features, heads, key_size)
    for name in ("wq", "bq", "wk", "bk"):
        c[name] = np.zeros_like(c[name])
    out = _forward(c, heads)
    assert np.all(out["m"] == 0.0) and np.all(out["l"] == 16.0)
    for n in range(examples):
        v_n = out["v"][n * tokens : (n + 1) * tokens]
        total = to_numpy(to_array(np.ones((1, tokens))) @ to_array(v_n))[0]
        for row in out["h"][n * tokens : (n + 1) * tokens]:
            assert bits(row) == bits(total / 16.0)


def _repeated(block: np.ndarray, heads: int) -> np.ndarray:
    # one head's rows of a weight or bias, as every head's
    return np.concatenate([block] * heads)


@pytest.mark.parametrize("examples,tokens,features,heads,key_size", MULTI_HEAD, ids=map(_id, MULTI_HEAD))
def test_identical_heads_have_identical_outputs(examples: int, tokens: int, features: int, heads: int, key_size: int):
    c = _case(examples, tokens, features, heads, key_size)
    for name in ("wq", "bq", "wk", "bk", "wv", "bv"):
        c[name] = _repeated(c[name][:key_size], heads)
    out = _forward(c, heads)
    for i in range(1, heads):
        assert bits(out["m"][:, i]) == bits(out["m"][:, 0]) and bits(out["l"][:, i]) == bits(out["l"][:, 0])
        assert bits(out["h"][:, i * key_size : (i + 1) * key_size]) == bits(out["h"][:, :key_size])


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
        assert not np.any(b[name][:, silent]), name
        assert np.any(b[name][:, key_size:] != 0.0), name
    grads = dict(zip(NAMES, _gradients(c, f, b)))
    for name in ("wq", "bq", "wk", "bk", "wv", "bv"):
        assert not np.any(grads[name][silent]), name


@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_no_later_token_moves_an_earlier_output_of_a_causal_pass(
    examples: int, tokens: int, features: int, heads: int, key_size: int
):
    # token t's rows of a and h read only tokens 0..t: a later key's weight is exactly 0
    c = _case(examples, tokens, features, heads, key_size)
    out = _forward(c, heads, causal=True)
    for t in sorted({0, tokens // 2, tokens - 2}):
        if t < 0 or t >= tokens - 1:
            continue
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
    for name in ["a", "q", "k", "v", "h", "m", "l"]:
        assert bits(default[name]) == bits(unmasked[name]), name
    # token 0 weights only itself: e = 1, l = 1, so its row of H is its row of V exactly
    for n in range(examples):
        assert bits(masked["h"][n * tokens]) == bits(masked["v"][n * tokens])
    if tokens == 1:
        for name in ["a", "q", "k", "v", "h", "m", "l"]:
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
    with pytest.raises(ValueError, match="attention_forward requires key blocks of at least one key"):
        attention_forward(Array([0.0] * 12), *parameters, heads=2, block=0)
    for heads in (0, 3):
        with pytest.raises(ValueError, match=rf"heads >= 1 dividing the projections' width 4, got heads={heads}"):
            attention_forward(Array([0.0] * 12), *parameters, heads=heads)
    wq, bq, wk, bk, wv, bv, wo, bo = parameters
    with pytest.raises(ValueError, match=r"requires wo of shape Matrix\(6, 4\)"):
        attention_forward(Array([0.0] * 12), wq, bq, wk, bk, wv, bv, wq, bo, 2)
    with pytest.raises(ValueError, match=r"requires bk of shape Vector\(4\)"):
        attention_forward(Array([0.0] * 12), wq, bq, wk, bo, wv, bv, wo, bo, 2)
    f = _forward(c, 2)
    q, k, v, h, m, l = (to_array(f[name]) for name in ("q", "k", "v", "h", "m", "l"))
    delta = to_array(c["delta"])
    with pytest.raises(ValueError, match=r"attention_downstream_batch requires m of shape Matrix\(2, 2\)"):
        attention_downstream_batch(delta, wq, wk, wv, wo, q, k, v, h, to_array(f["m"][:, :1]), l, 2)
    with pytest.raises(ValueError, match=r"attention_downstream_batch requires h of shape Matrix\(2, 4\)"):
        attention_downstream_batch(delta, wq, wk, wv, wo, q, k, v, to_array(f["h"][:1]), m, l, 2)
    with pytest.raises(ValueError, match=r"attention_downstream_batch requires wo of shape Matrix\(6, 4\)"):
        attention_downstream_batch(delta, wq, wk, wv, wq, q, k, v, h, m, l, 2)
    with pytest.raises(ValueError, match="attention_downstream_batch requires key blocks of at least one key"):
        attention_downstream_batch(delta, wq, wk, wv, wo, q, k, v, h, m, l, 2, block=0)


@pytest.mark.parametrize(
    "tokens,features,heads,key_size", [(1, 3, 1, 3), (5, 6, 1, 6), (16, 32, 1, 32), (5, 6, 3, 2), (37, 8, 2, 4)]
)
def test_one_examples_backward_is_a_batch_of_ones(tokens: int, features: int, heads: int, key_size: int):
    c = _case(1, tokens, features, heads, key_size)
    x, delta = c["x"][0], c["delta"][0]
    _, q, k, v, h, m, l, _ = attention_forward(to_array(x), *_parameters(c), heads=heads, causal=True)
    weights = [to_array(c[name]) for name in ("wq", "wk", "wv", "wo")]
    wq, wk, wv, wo = weights
    single = attention_downstream_batch(to_array(delta), wq, wk, wv, wo, q, k, v, h, m, l, heads, causal=True)
    f = _forward(c, heads, causal=True)
    batch = _backward(c, f, heads, causal=True)
    assert single[0].shape == (tokens * features,)
    assert bits(to_numpy(single[0])) == bits(batch["dx"])
    for array, name in zip(single[1:], ["dq", "dk", "dv"]):
        assert bits(to_numpy(array)) == bits(batch[name]), name
    grads = [to_array(c["grad_" + name]) for name in NAMES]
    one = attention_accumulate_gradient_batch(to_array(delta), to_array(x), h, single[1], single[2], single[3], *grads)
    many = _gradients(c, f, batch)
    for i, name in enumerate(NAMES):
        assert bits(to_numpy(one[i])) == bits(many[i]), name


# dropout on the weights (the attention-dropout workplan; the context-length workplan, D12)

DROPOUT = 0.3


def _block_masks(examples: int, heads: int, tokens: int, seed: int, block: int = BLOCK) -> list[np.ndarray]:
    # numpy's draws key block by key block, each (N, h, T, b)
    rng = np.random.default_rng(seed)
    return [
        (rng.random((examples, heads, tokens, k1 - k0)) >= DROPOUT).astype(np.float64)
        for k0, k1 in key_blocks(tokens, block)
    ]


def _dropping(c: dict[str, np.ndarray], heads: int, causal: bool, seed: int, one: bool = False) -> dict[str, Any]:
    forward = attention_forward if one else attention_forward_batch
    x = c["x"][0] if one else c["x"]
    rng = Generator(seed)
    result = forward(to_array(x), *_parameters(c), heads=heads, causal=causal, dropout=DROPOUT, training=True, rng=rng)
    out: dict[str, Any] = {name: (to_numpy(r) if isinstance(r, Array) else r) for name, r in zip(FORWARD, result)}
    out["after"] = rng.state
    return out


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_without_training_or_at_dropout_0_nothing_is_drawn_and_the_bits_are_as_without_dropout(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    c = _case(examples, tokens, features, heads, key_size)
    before = _forward(c, heads, causal)
    plain = _backward(c, before, heads, causal)
    for dropout, training in [(DROPOUT, False), (0.0, True)]:
        rng = Generator(4)
        state = rng.state
        out = _forward(c, heads, causal, dropout=dropout, training=training, rng=rng)
        assert out["rng"] is None and rng.state == state
        for name in ["a", "q", "k", "v", "h", "m", "l"]:
            assert bits(out[name]) == bits(before[name]), name
        b = _backward(c, out, heads, causal, dropout=dropout)
        for name in ["dx", "dq", "dk", "dv"]:
            assert bits(b[name]) == bits(plain[name]), name


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_a_dropping_pass_is_the_composition_with_numpys_key_block_draws(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    c = _case(examples, tokens, features, heads, key_size)
    out = _dropping(c, heads, causal, seed=7)
    masks = _block_masks(examples, heads, tokens, 7)
    keep = 1.0 - DROPOUT
    composed = _composed_forward(c, tokens, heads, causal, masks=masks, keep=keep)
    for name in ["a", "q", "k", "v", "h", "m", "l"]:
        assert bits(out[name]) == bits(composed[name]), name
    # exactly N * h * T * T draws: the generator is where numpy's draws leave it, and the forward's
    # copy is where it started
    after = np.random.default_rng(7)
    after.random(examples * heads * tokens * tokens)
    assert out["after"] == after.bit_generator.state
    assert out["rng"].state == Generator(7).state
    b = _backward(c, out, heads, causal, dropout=DROPOUT, dropout_rng=out["rng"])
    composed_b = _composed_backward(c, out, tokens, heads, causal, masks=masks, keep=keep)
    for name in ["dx", "dq", "dk", "dv"]:
        assert bits(b[name]) == bits(composed_b[name]), name
    # the backward drew from a copy: the forward's copy is unmoved, so a second backward is the same
    assert out["rng"].state == Generator(7).state
    again = _backward(c, out, heads, causal, dropout=DROPOUT, dropout_rng=out["rng"])
    assert [bits(again[n]) for n in again] == [bits(b[n]) for n in b]


@pytest.mark.parametrize("examples,tokens,features,heads,key_size", SHAPES, ids=map(_id, SHAPES))
def test_one_examples_dropping_pass_is_its_batch_of_ones(
    examples: int, tokens: int, features: int, heads: int, key_size: int
):
    c = _case(1, tokens, features, heads, key_size)
    one, batch = _dropping(c, heads, True, seed=3, one=True), _dropping(c, heads, True, seed=3)
    assert bits(one["a"]) == bits(batch["a"][0])
    for name in ["h", "m", "l"]:
        assert bits(one[name]) == bits(batch[name]), name
    assert one["after"] == batch["after"]


def test_the_dropping_ops_refuse_a_dropout_outside_0_to_1_and_a_missing_rng():
    c = _case(2, 3, 4, 2, 2)
    x = to_array(c["x"])
    for dropout in (-0.1, 1.0, math.nan):
        with pytest.raises(ValueError, match=r"attention_forward_batch requires a dropout in \[0, 1\)"):
            attention_forward_batch(x, *_parameters(c), heads=2, dropout=dropout)
    with pytest.raises(ValueError, match="attention_forward requires an rng to drop out in training"):
        attention_forward(to_array(c["x"][0]), *_parameters(c), heads=2, dropout=0.1, training=True)
    f = _dropping(c, 2, False, seed=1)
    with pytest.raises(ValueError, match=r"attention_downstream_batch requires a dropout in \[0, 1\)"):
        _backward(c, f, 2, dropout=1.0, dropout_rng=f["rng"])


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
    return {name: to_numpy(array) for name, array in zip(FORWARD[:7], result[:7]) if isinstance(array, Array)}


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("examples,tokens,features,heads,key_size", ROTARY, ids=map(_id, ROTARY))
def test_a_rotary_forward_pass_is_the_composition_with_q_and_k_rotated_after_project(
    examples: int, tokens: int, features: int, heads: int, key_size: int, causal: bool
):
    c = _case(examples, tokens, features, heads, key_size)
    table = _table(tokens, key_size)
    q, k, v = _projected(c)
    rotated = (_rotated(q, table, heads), _rotated(k, table, heads), v)
    out, composed = _rotary_forward(c, tokens, heads, causal), _composed_forward(c, tokens, heads, causal, qkv=rotated)
    for name in ["a", "q", "k", "v", "h", "m", "l"]:
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
    out = _backward(c, f, heads, causal, cos=to_array(table[0]), sin=to_array(table[1]))
    # the unrotated op's attend gradients from the rotated caches, then rotated back, then dx
    plain = _backward(c, f, heads, causal)
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
    for name in ["q", "k", "v", "h", "m", "l"]:
        assert bits(one[name]) == bits(batch[name][:tokens]), name


def test_position_0_is_unrotated_so_one_token_is_the_unrotated_pass_by_bits():
    c = _case(2, 1, 6, 3, 2)
    f, plain = _rotary_forward(c, 1, 3, False), _forward(c, 3)
    for name in ["a", "q", "k", "v", "h", "m", "l"]:
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
    with pytest.raises(ValueError, match="attention_downstream_batch requires both cos and sin to rotate, or neither"):
        _backward(c, f, 3, sin=sin)
    with pytest.raises(ValueError, match=r"attention_downstream_batch requires cos of shape Matrix\(3, 1\)"):
        _backward(c, f, 3, cos=Array([[1.0]] * 2), sin=sin)
