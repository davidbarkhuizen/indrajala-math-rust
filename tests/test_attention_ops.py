"""
The attention ops (`attention_*`, indrajala-ml's layer-norm and attention workplan, D6 and D9): one
call per pass. Each is checked by bits against the crate's own unfused composition of indrajala-ml's
README expressions (Layer norm and attention), built here from the crate's existing ops: the
projections, the downstream products and the gradients as a dense layer's (`affine_forward_batch`,
`layer_downstream_batch`, `layer_accumulate_gradient_batch`) on the `(N * T, d)` rows; the
products between activations as `@` per example; the row softmax as `array_softmax`; the softmax
backward's row sums as `sum_axis0` of the transpose, a left fold. The elementwise steps between
them are numpy's, correctly rounded in the README's grouping. numpy's own products (BLAS) are not
the crate's, so numpy is the reference only where no product is involved, and in the exact tests.
"""

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

# (examples, tokens, features): one token, the patch model's 16 x 32, and widths off the 4 lanes
SHAPES = [(1, 1, 1), (2, 1, 3), (1, 4, 3), (3, 5, 6), (2, 16, 32), (2, 7, 33)]
NAMES = ["wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo"]


def _numpy(array: Array) -> np.ndarray:
    return np.array(array.tolist())


def _array(values: np.ndarray) -> Array:
    return Array(values.tolist())


def _bits(values: np.ndarray) -> list[bytes]:
    # by bits, so -0.0 and 0.0 differ, where np.array_equal takes them as equal
    return [v.tobytes() for v in np.ravel(values).astype(np.float64)]


def _case(examples: int, tokens: int, features: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(examples * 10000 + tokens * 100 + features)
    d, limit = features, 1.0 / math.sqrt(features)
    c = {"x": rng.uniform(-2.0, 2.0, (examples, tokens * d))}
    for name in NAMES:
        c[name] = rng.uniform(-limit, limit, (d, d) if name.startswith("w") else d)
    c["delta"] = rng.uniform(-1.0, 1.0, (examples, tokens * d))
    for name in NAMES:
        c["grad_" + name] = rng.uniform(-1.0, 1.0, c[name].shape)
    return c


def _parameters(c: dict[str, np.ndarray]) -> list[Array]:
    return [_array(c[name]) for name in NAMES]


def _block(rows: np.ndarray, n: int, tokens: int) -> Array:
    return _array(rows[n * tokens : (n + 1) * tokens])


def _composed_forward(c: dict[str, np.ndarray], tokens: int) -> dict[str, np.ndarray]:
    d = c["bq"].shape[0]
    rows = _array(c["x"].reshape(-1, d))
    q, k, v = (_numpy(affine_forward_batch(_array(c["w" + n]), rows, _array(c["b" + n]))) for n in ("q", "k", "v"))
    p, h = [], []
    for n in range(c["x"].shape[0]):
        q_n, k_n, v_n = _block(q, n, tokens), _block(k, n, tokens), _block(v, n, tokens)
        p_n = array_softmax((q_n @ k_n.T) / math.sqrt(d))
        p.append(_numpy(p_n))
        h.append(_numpy(p_n @ v_n))
    h_rows = np.concatenate(h)
    a = _numpy(affine_forward_batch(_array(c["wo"]), _array(h_rows), _array(c["bo"])))
    return {"a": a.reshape(c["x"].shape), "q": q, "k": k, "v": v, "p": np.concatenate(p), "h": h_rows}


def _forward(c: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = attention_forward_batch(_array(c["x"]), *_parameters(c))
    return {name: _numpy(array) for name, array in zip(["a", "q", "k", "v", "p", "h"], result)}


def _composed_backward(c: dict[str, np.ndarray], f: dict[str, np.ndarray], tokens: int) -> dict[str, np.ndarray]:
    d = c["bq"].shape[0]
    s = math.sqrt(d)
    dh = _numpy(layer_downstream_batch(_array(c["wo"]), _array(c["delta"].reshape(-1, d))))
    dq, dk, dv = [], [], []
    for n in range(c["x"].shape[0]):
        q_n, k_n, v_n, p_n = (_block(f[name], n, tokens) for name in ("q", "k", "v", "p"))
        dh_n = _block(dh, n, tokens)
        dp = _numpy(dh_n @ v_n.T)
        dv.append(_numpy(p_n.T @ dh_n))
        p = _numpy(p_n)
        r = _numpy(sum_axis0(_array((dp * p).T)))  # each row's left fold
        ds = _array(p * (dp - r[:, np.newaxis]))
        dq.append(_numpy((ds @ k_n) / s))
        dk.append(_numpy((ds.T @ q_n) / s))
    dq_rows, dk_rows, dv_rows = np.concatenate(dq), np.concatenate(dk), np.concatenate(dv)
    dx = (
        layer_downstream_batch(_array(c["wq"]), _array(dq_rows))
        + layer_downstream_batch(_array(c["wk"]), _array(dk_rows))
    ) + layer_downstream_batch(_array(c["wv"]), _array(dv_rows))
    return {"dx": _numpy(dx).reshape(c["x"].shape), "dq": dq_rows, "dk": dk_rows, "dv": dv_rows}


def _backward(c: dict[str, np.ndarray], f: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = attention_downstream_batch(
        _array(c["delta"]),
        *(_array(c[name]) for name in ("wq", "wk", "wv", "wo")),
        *(_array(f[name]) for name in ("q", "k", "v", "p")),
    )
    return {name: _numpy(array) for name, array in zip(["dx", "dq", "dk", "dv"], result)}


@pytest.mark.parametrize("examples,tokens,features", SHAPES)
def test_the_forward_pass_is_the_crates_composition(examples: int, tokens: int, features: int):
    c = _case(examples, tokens, features)
    out, composed = _forward(c), _composed_forward(c, tokens)
    for name in ["a", "q", "k", "v", "p", "h"]:
        assert out[name].shape == composed[name].shape, name
        assert _bits(out[name]) == _bits(composed[name]), name


@pytest.mark.parametrize("examples,tokens,features", SHAPES)
def test_one_examples_forward_pass_is_its_batch_rows(examples: int, tokens: int, features: int):
    c = _case(examples, tokens, features)
    batch = _forward(c)
    for i in range(examples):
        result = attention_forward(_array(c["x"][i]), *_parameters(c))
        single = {name: _numpy(array) for name, array in zip(["a", "q", "k", "v", "p", "h"], result)}
        assert _bits(single["a"]) == _bits(batch["a"][i])
        for name in ["q", "k", "v", "p", "h"]:
            assert _bits(single[name]) == _bits(batch[name][i * tokens : (i + 1) * tokens]), name


@pytest.mark.parametrize("examples,tokens,features", SHAPES)
def test_the_downstream_is_the_crates_composition(examples: int, tokens: int, features: int):
    c = _case(examples, tokens, features)
    f = _forward(c)
    out, composed = _backward(c, f), _composed_backward(c, f, tokens)
    for name in ["dx", "dq", "dk", "dv"]:
        assert out[name].shape == composed[name].shape, name
        assert _bits(out[name]) == _bits(composed[name]), name


@pytest.mark.parametrize("examples,tokens,features", SHAPES)
def test_the_gradients_are_each_projections_dense_gradients(examples: int, tokens: int, features: int):
    c = _case(examples, tokens, features)
    f = _forward(c)
    b = _backward(c, f)
    grads = attention_accumulate_gradient_batch(
        _array(c["delta"]),
        _array(c["x"]),
        _array(f["h"]),
        _array(b["dq"]),
        _array(b["dk"]),
        _array(b["dv"]),
        *(_array(c["grad_" + name]) for name in NAMES),
    )
    rows, delta = _array(c["x"].reshape(-1, features)), _array(c["delta"].reshape(-1, features))
    inputs = {"q": (b["dq"], rows), "k": (b["dk"], rows), "v": (b["dv"], rows), "o": (None, _array(f["h"]))}
    for i, n in enumerate("qkvo"):
        d_rows, x_rows = inputs[n]
        expected = layer_accumulate_gradient_batch(
            delta if d_rows is None else _array(d_rows), x_rows, _array(c["grad_w" + n]), _array(c["grad_b" + n])
        )
        assert _bits(_numpy(grads[2 * i])) == _bits(_numpy(expected[0])), "w" + n
        assert _bits(_numpy(grads[2 * i + 1])) == _bits(_numpy(expected[1])), "b" + n


@pytest.mark.parametrize("examples,features", [(1, 1), (3, 5), (2, 32)])
def test_with_one_token_attention_is_the_value_then_the_output_projection(examples: int, features: int):
    # P = [[1]], so out = (X Wv^T + bv) Wo^T + bo by bits
    c = _case(examples, 1, features)
    out = _forward(c)
    assert np.all(out["p"] == 1.0)
    v = affine_forward_batch(_array(c["wv"]), _array(c["x"]), _array(c["bv"]))
    expected = affine_forward_batch(_array(c["wo"]), v, _array(c["bo"]))
    assert _bits(out["a"]) == _bits(_numpy(expected))


@pytest.mark.parametrize("examples,features", [(1, 1), (2, 7), (2, 32)])
def test_with_zero_query_and_key_weights_attention_is_uniform_and_h_the_token_mean_of_v(examples: int, features: int):
    # every score is 0 and every weight exactly 1/16, so each row of H is the mean of V's rows,
    # 1/16 * v exact and the product's sum a left fold, as the token mean's
    tokens = 16
    c = _case(examples, tokens, features)
    for name in ("wq", "bq", "wk", "bk"):
        c[name] = np.zeros_like(c[name])
    out = _forward(c)
    assert np.all(out["p"] == 1.0 / 16)
    for n in range(examples):
        v_n = out["v"][n * tokens : (n + 1) * tokens]
        mean = _numpy(token_mean_forward(_array(v_n.ravel()), tokens))
        for row in out["h"][n * tokens : (n + 1) * tokens]:
            assert _bits(row) == _bits(mean)


def test_the_attention_ops_refuse_shapes_that_dont_fit():
    c = _case(1, 2, 3)
    parameters = _parameters(c)
    with pytest.raises(ValueError, match=r"attention_forward_batch requires x of whole tokens of 3 features"):
        attention_forward_batch(Array([[0.0] * 4]), *parameters)
    with pytest.raises(ValueError, match="attention_forward_batch requires a 2D x"):
        attention_forward_batch(Array([0.0] * 6), *parameters)
    with pytest.raises(ValueError, match="attention_forward requires a 1D x"):
        attention_forward(Array([[0.0] * 6]), *parameters)
    with pytest.raises(ValueError, match=r"requires every weight of shape \(3, 3\) and every bias of shape \(3,\)"):
        wq, bq, wk, bk, wv, bv, wo, _ = parameters
        attention_forward(Array([0.0] * 6), wq, bq, wk, bk, wv, bv, wo, Array([0.0] * 2))
    f = _forward(c)
    wq, wk, wv, wo = (_array(c[name]) for name in ("wq", "wk", "wv", "wo"))
    q, k, v = (_array(f[name]) for name in ("q", "k", "v"))
    with pytest.raises(ValueError, match="attention_downstream_batch requires p of shape"):
        attention_downstream_batch(_array(c["delta"]), wq, wk, wv, wo, q, k, v, q)
