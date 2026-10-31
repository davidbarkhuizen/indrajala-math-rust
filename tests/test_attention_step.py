"""
The decode step (`attention_step`, indrajala-ml's generation workplan, D2, D3, D8): one token's
causal attention on the cache of the tokens before it. Its blocks are the forward pass's, a score one
chain over `d_k` and a weighted sum one chain over the keys in order, and a causal pass's masked
keys add exact zeros, so a step at token `t` is the causal forward pass's row `t` by bits, its cache
the pass's keys and values up to `t`, with the queries and keys rotated by the table's row `t`. Past
the window the caches keep the last `window` tokens', oldest first.
"""

import math

import numpy as np
import pytest
from array_helpers import bits, to_array, to_numpy

import indrajala_math_rust as pa
from indrajala_math_rust import Array, attention_forward, attention_step, set_matmul_threading

# (tokens, features, heads, key_size)
# the sequence study's 64 tokens of 64 in 4 heads among them, past the products' blocking
SHAPES = [(1, 3, 1, 3), (5, 6, 1, 6), (7, 12, 4, 4), (16, 32, 4, 8), (9, 33, 3, 6), (6, 8, 2, 2), (64, 64, 4, 16)]
NAMES = ["wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo"]


def _case(tokens: int, features: int, heads: int, key_size: int) -> dict[str, np.ndarray]:
    # one example, from the crate's own PCG64, as test_attention_ops' cases
    rng = pa.default_rng([tokens, features, heads, key_size, 9])

    def uniform(low: float, high: float, shape: tuple[int, int] | int) -> np.ndarray:
        return to_numpy(rng.uniform(low, high, shape))

    d, width = features, heads * key_size
    c = {"x": uniform(-2.0, 2.0, tokens * d)}
    for name in NAMES:
        rows, fan_in = (d, width) if name[1] == "o" else (width, d)
        limit = 1.0 / math.sqrt(fan_in)
        c[name] = uniform(-limit, limit, (rows, fan_in) if name[0] == "w" else rows)
    return c


def _parameters(c: dict[str, np.ndarray]) -> list[Array]:
    return [to_array(c[name]) for name in NAMES]


def _table(tokens: int, key_size: int, base: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    # indrajala-ml's rotary_table: math's cos and sin of t * theta_j, (T, d_k / 2) each
    angles = [[t * base ** (-2 * j / key_size) for j in range(key_size // 2)] for t in range(tokens)]
    cos = np.array([[math.cos(a) for a in row] for row in angles]).reshape(tokens, key_size // 2)
    sin = np.array([[math.sin(a) for a in row] for row in angles]).reshape(tokens, key_size // 2)
    return cos, sin


def _forward(c: dict[str, np.ndarray], heads: int, rotary: bool) -> dict[str, np.ndarray]:
    tokens = c["x"].shape[0] // c["bo"].shape[0]
    table = {}
    if rotary:
        cos, sin = _table(tokens, c["bq"].shape[0] // heads)
        table = {"cos": to_array(cos), "sin": to_array(sin)}
    result = attention_forward(to_array(c["x"]), *_parameters(c), heads=heads, causal=True, **table)
    return {name: to_numpy(array) for name, array in zip(["a", "q", "k", "v"], result[:4])}


def _step(
    c: dict[str, np.ndarray], t: int, k_cache: np.ndarray, v_cache: np.ndarray, heads: int, window: int, rotary: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d, width = c["bo"].shape[0], c["bq"].shape[0]
    table = {}
    if rotary:
        cos, sin = _table(t + 1, width // heads)
        table = {"cos": to_array(cos[t : t + 1]), "sin": to_array(sin[t : t + 1])}
    x = to_array(c["x"][t * d : (t + 1) * d])
    caches = (Array.zeros((0, width)), Array.zeros((0, width))) if t == 0 else (to_array(k_cache), to_array(v_cache))
    a, k, v = attention_step(
        x, *_parameters(c), k_cache=caches[0], v_cache=caches[1], heads=heads, window=window, **table
    )
    return to_numpy(a), to_numpy(k).reshape(-1, width), to_numpy(v).reshape(-1, width)


@pytest.mark.parametrize("rotary", [False, True])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_each_step_is_the_causal_forward_pass_at_its_token_by_bits(shape: tuple[int, ...], rotary: bool) -> None:
    tokens, features, heads, key_size = shape
    if rotary and key_size % 2:
        pytest.skip("a rotary head is even")
    c = _case(*shape)
    f = _forward(c, heads, rotary)
    d = features
    for t in range(tokens):
        a, k, v = _step(c, t, f["k"][:t], f["v"][:t], heads, tokens, rotary)
        assert bits(a) == bits(f["a"][t * d : (t + 1) * d]), t
        assert bits(k) == bits(f["k"][: t + 1]), t
        assert bits(v) == bits(f["v"][: t + 1]), t


def test_a_threaded_forward_pass_is_the_steps_bits() -> None:
    # every product split across 8 threads (threshold 1 flop): each output row's chain is the same
    tokens, features, heads, key_size = 64, 64, 4, 16
    c = _case(tokens, features, heads, key_size)
    set_matmul_threading(8, 1)
    try:
        f = _forward(c, heads, True)
    finally:
        set_matmul_threading(0, 0)
    for t in (0, 31, 63):
        a, _k, _v = _step(c, t, f["k"][:t], f["v"][:t], heads, tokens, True)
        assert bits(a) == bits(f["a"][t * features : (t + 1) * features]), t


@pytest.mark.parametrize("rotary", [False, True])
def test_past_the_window_the_oldest_key_and_value_are_dropped(rotary: bool) -> None:
    tokens, features, heads, key_size = 9, 12, 2, 4
    c = _case(tokens, features, heads, key_size)
    f = _forward(c, heads, rotary)
    window, t = 4, 8
    k_cache, v_cache = f["k"][t - window : t], f["v"][t - window : t]
    a, k, v = _step(c, t, k_cache, v_cache, heads, window, rotary)
    assert bits(k) == bits(f["k"][t - window + 1 : t + 1])
    assert bits(v) == bits(f["v"][t - window + 1 : t + 1])
    # the same as a step on the window's last keys only: the dropped one isn't read
    trimmed, _k, _v = _step(c, t, k_cache[1:], v_cache[1:], heads, window, rotary)
    assert bits(a) == bits(trimmed)
    assert bits(a) != bits(f["a"][t * features : (t + 1) * features])


def test_the_step_refuses_shapes_that_dont_fit() -> None:
    c = _case(3, 6, 2, 2)
    parameters = _parameters(c)
    x, cache = to_array(c["x"][:6]), Array.zeros((2, 4))
    with pytest.raises(ValueError, match=r"attention_step requires x of shape Vector\(6\)"):
        attention_step(to_array(c["x"][:5]), *parameters, k_cache=cache, v_cache=cache, heads=2, window=3)
    with pytest.raises(ValueError, match=r"attention_step requires k_cache of shape \(c, 4\)"):
        attention_step(x, *parameters, k_cache=Array.zeros((2, 3)), v_cache=cache, heads=2, window=3)
    with pytest.raises(ValueError, match=r"attention_step requires v_cache of shape Matrix\(2, 4\)"):
        attention_step(x, *parameters, k_cache=cache, v_cache=Array.zeros((1, 4)), heads=2, window=3)
    with pytest.raises(ValueError, match="attention_step requires a window of at least one token"):
        attention_step(x, *parameters, k_cache=cache, v_cache=cache, heads=2, window=0)
    cos = to_array(_table(1, 2)[0][:1])
    with pytest.raises(ValueError, match="attention_step requires both cos and sin to rotate, or neither"):
        attention_step(x, *parameters, k_cache=cache, v_cache=cache, heads=2, window=3, cos=cos)
    with pytest.raises(ValueError, match=r"attention_step requires sin of shape Matrix\(1, 1\)"):
        attention_step(x, *parameters, k_cache=cache, v_cache=cache, heads=2, window=3, cos=cos, sin=Array([[0.0]] * 2))
