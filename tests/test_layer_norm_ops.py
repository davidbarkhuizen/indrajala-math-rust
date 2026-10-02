"""
The layer-norm ops (`layer_norm_*`, indrajala-ml's layer-norm and attention workplan, D5) against
numpy transcriptions of indrajala-ml's README expressions (Layer norm and attention), as its
`LayerNormArrayLayer` computes them: on the `(N * T, d)` rows, every sum a left fold through
`np.cumsum`. The ops use only correctly rounded `+ - * /` and `sqrt`, so they are compared by bits.
"""

import numpy as np
import pytest

from indrajala_math_rust import (
    Array,
    layer_norm_accumulate_gradient_batch,
    layer_norm_downstream_batch,
    layer_norm_forward,
    layer_norm_forward_batch,
)

EPSILON = 1e-5
# (examples, tokens, features): a flat layer is one token; 33 features cross numpy's pairwise block
SHAPES = [(1, 1, 1), (1, 1, 7), (2, 1, 33), (3, 4, 5), (2, 16, 32), (5, 3, 2)]


def _numpy(array: Array) -> np.ndarray:
    return np.array(array.tolist())


def _bits(values: np.ndarray) -> list[bytes]:
    # by bits, so -0.0 and 0.0 differ, where np.array_equal takes them as equal
    return [v.tobytes() for v in np.ravel(values).astype(np.float64)]


def _sum_features(rows: np.ndarray) -> np.ndarray:
    # LayerNormArrayLayer's sum_features: each row's left fold, as a column
    return np.cumsum(rows, axis=1)[:, -1:]


def _case(examples: int, tokens: int, features: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(examples * 10000 + tokens * 100 + features)
    width = tokens * features
    return {
        "x": rng.uniform(-3.0, 3.0, (examples, width)),
        "gamma": rng.uniform(0.5, 1.5, features),
        "beta": rng.uniform(-0.5, 0.5, features),
        "delta": rng.uniform(-1.0, 1.0, (examples, width)),
        "grad_gamma": rng.uniform(-1.0, 1.0, features),
        "grad_beta": rng.uniform(-1.0, 1.0, features),
    }


def _forward_reference(c: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    d = c["gamma"].shape[0]
    rows = c["x"].reshape(-1, d)
    mu = _sum_features(rows) / d
    centered = rows - mu
    var = _sum_features(centered * centered) / d
    std = np.sqrt(var + EPSILON)
    xhat = centered / std
    return {
        "a": (c["gamma"] * xhat + c["beta"]).reshape(c["x"].shape),
        "xhat": xhat.reshape(c["x"].shape),
        "std": std.ravel(),
    }


def _forward(c: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = layer_norm_forward_batch(
        Array(c["x"].tolist()), Array(c["gamma"].tolist()), Array(c["beta"].tolist()), EPSILON
    )
    return {name: _numpy(array) for name, array in zip(["a", "xhat", "std"], result)}


@pytest.mark.parametrize("examples,tokens,features", SHAPES)
def test_the_forward_pass_is_the_readmes_expressions(examples: int, tokens: int, features: int):
    c = _case(examples, tokens, features)
    out, reference = _forward(c), _forward_reference(c)
    for name in ["a", "xhat", "std"]:
        assert out[name].shape == reference[name].shape, name
        assert _bits(out[name]) == _bits(reference[name]), name


@pytest.mark.parametrize("examples,tokens,features", SHAPES)
def test_one_examples_forward_pass_is_its_batch_rows(examples: int, tokens: int, features: int):
    c = _case(examples, tokens, features)
    batch = _forward(c)
    for i in range(examples):
        a, xhat, std = layer_norm_forward(
            Array(c["x"][i].tolist()), Array(c["gamma"].tolist()), Array(c["beta"].tolist()), EPSILON
        )
        assert _bits(_numpy(a)) == _bits(batch["a"][i])
        assert _bits(_numpy(xhat)) == _bits(batch["xhat"][i])
        assert _bits(_numpy(std)) == _bits(batch["std"][i * tokens : (i + 1) * tokens])


@pytest.mark.parametrize("examples,tokens,features", SHAPES)
def test_the_downstream_is_the_readmes_expression(examples: int, tokens: int, features: int):
    c = _case(examples, tokens, features)
    t = _forward_reference(c)
    d = features
    xhat, std = t["xhat"].reshape(-1, d), t["std"].reshape(-1, 1)
    dxhat = c["delta"].reshape(-1, d) * c["gamma"]
    m1 = _sum_features(dxhat) / d
    m2 = _sum_features(dxhat * xhat) / d
    reference = (((dxhat - m1) - xhat * m2) / std).reshape(c["x"].shape)
    dx = layer_norm_downstream_batch(
        Array(c["delta"].tolist()), Array(c["gamma"].tolist()), Array(t["xhat"].tolist()), Array(t["std"].tolist())
    )
    assert _bits(_numpy(dx)) == _bits(reference)


@pytest.mark.parametrize("examples,tokens,features", SHAPES)
def test_the_gradients_are_left_folds_over_the_rows(examples: int, tokens: int, features: int):
    c = _case(examples, tokens, features)
    xhat = _forward_reference(c)["xhat"]
    delta, rows = c["delta"].reshape(-1, features), xhat.reshape(-1, features)
    grad_gamma, grad_beta = layer_norm_accumulate_gradient_batch(
        Array(c["delta"].tolist()),
        Array(xhat.tolist()),
        Array(c["grad_gamma"].tolist()),
        Array(c["grad_beta"].tolist()),
    )
    assert _bits(_numpy(grad_gamma)) == _bits(c["grad_gamma"] + np.cumsum(delta * rows, axis=0)[-1])
    assert _bits(_numpy(grad_beta)) == _bits(c["grad_beta"] + np.cumsum(delta, axis=0)[-1])


def test_a_constant_token_normalizes_to_beta():
    # var = 0, so xhat = 0 / sqrt(eps) = 0 and y = beta exactly
    gamma, beta = Array([2.0, 3.0]), Array([0.25, -0.5])
    a, xhat, std = layer_norm_forward_batch(Array([[4.0, 4.0, -1.0, -1.0]]), gamma, beta, EPSILON)
    assert a.tolist() == [[0.25, -0.5, 0.25, -0.5]]
    assert xhat.tolist() == [[0.0, 0.0, 0.0, 0.0]]
    assert std.tolist() == [np.sqrt(EPSILON)] * 2


def test_the_layer_norm_ops_refuse_shapes_that_dont_fit():
    gamma, beta = Array([1.0, 1.0, 1.0]), Array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"layer_norm_forward_batch requires x of whole tokens of 3 features"):
        layer_norm_forward_batch(Array([[0.0] * 4]), gamma, beta, EPSILON)
    with pytest.raises(ValueError, match="layer_norm_forward_batch requires a 2D x"):
        layer_norm_forward_batch(Array([0.0] * 3), gamma, beta, EPSILON)
    with pytest.raises(ValueError, match="layer_norm_forward requires a 1D x"):
        layer_norm_forward(Array([[0.0] * 3]), gamma, beta, EPSILON)
    with pytest.raises(ValueError, match="layer_norm_forward requires beta of shape"):
        layer_norm_forward(Array([0.0] * 3), gamma, Array([0.0]), EPSILON)
    delta = Array([[0.0] * 6])
    with pytest.raises(ValueError, match="layer_norm_downstream_batch requires std of shape"):
        layer_norm_downstream_batch(delta, gamma, delta, Array([1.0]))
    with pytest.raises(ValueError, match="layer_norm_accumulate_gradient_batch requires xhat of shape"):
        layer_norm_accumulate_gradient_batch(delta, Array([[0.0] * 3]), gamma, beta)
