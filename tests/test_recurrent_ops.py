"""
A window-recurrent network's ops (indrajala-ml's window-recurrent adder workplan, stage 5):
`state_input_*`, `attention_pooling_*`, `gru_*`, `codes_*` and `concatenate_columns`. Each is
checked by bits against its composition (src/recurrent.rs's expressions): the products are the
crate's existing ops (`linear_forward_batch`, `affine_forward_batch`, `layer_downstream_batch`,
`layer_accumulate_gradient_batch`, `linear_accumulate_gradient_batch`, `@`), the softmax
`array_softmax`, `exp` the crate's, and every other step numpy's, exact: the left folds as
`np.cumsum`, the elementwise expressions in the ops' grouping. The codes' ops have no product and
are numpy's own expressions (indrajala-ml's `CodesArrayLayer`) by bits. A case per op is over the
threading threshold, so its threaded products are covered.
"""

import numpy as np
import pytest

from indrajala_math_rust import (
    Array,
    affine_forward_batch,
    array_softmax,
    attention_pooling_backward,
    attention_pooling_forward,
    codes_backward,
    codes_forward,
    concatenate_columns,
    exp,
    gru_backward,
    gru_forward,
    layer_accumulate_gradient_batch,
    layer_downstream_batch,
    linear_accumulate_gradient_batch,
    linear_forward_batch,
    matmul_threads_for,
    state_input_backward,
    state_input_forward,
)

# (examples, tokens, features, state size); the last over the threading threshold
STATE_INPUTS = [(1, 1, 1, 1), (3, 4, 16, 4), (2, 5, 33, 16), (4096, 2, 128, 16)]
# (examples, tokens, features)
POOLINGS = [(1, 1, 1), (3, 4, 16), (2, 7, 33), (512, 6, 64)]
# (examples, input size, state size); the last over the threading threshold
GRUS = [(1, 1, 1), (3, 16, 4), (5, 33, 16), (2048, 64, 64)]
# (examples, count, size)
CODES = [(1, 1, 1), (5, 3, 4), (64, 16, 16), (7, 8, 33)]


def _numpy(array: Array) -> np.ndarray:
    return np.array(array.tolist())


def _array(values: np.ndarray) -> Array:
    return Array(values.tolist())


def _bits(values: np.ndarray) -> list[bytes]:
    # by bits, so -0.0 and 0.0 differ, where np.array_equal takes them as equal
    return [v.tobytes() for v in np.ravel(values).astype(np.float64)]


def _fold(values: np.ndarray, axis: int) -> np.ndarray:
    return np.take(np.cumsum(values, axis=axis), -1, axis=axis)


def _crate_exp(values: np.ndarray) -> np.ndarray:
    return _numpy(exp(_array(values))).reshape(values.shape)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + _crate_exp(-values))


def _tanh(values: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 / (_crate_exp(2.0 * values) + 1.0)


@pytest.mark.parametrize("n,tokens,d,s", STATE_INPUTS)
def test_state_input_forward_adds_the_projected_state_to_every_token(n: int, tokens: int, d: int, s: int):
    rng = np.random.default_rng(n + tokens + d + s)
    X, C, P = rng.uniform(-1, 1, (n, tokens * d)), rng.uniform(-1, 1, (n, s)), rng.uniform(-1, 1, (d, s))
    out = _numpy(state_input_forward(_array(X), _array(C), _array(P), tokens))
    projected = _numpy(linear_forward_batch(_array(P), _array(C)))
    expected = (X.reshape(n, tokens, d) + projected[:, np.newaxis, :]).reshape(n, -1)
    assert _bits(out) == _bits(expected)


@pytest.mark.parametrize("n,tokens,d,s", STATE_INPUTS)
def test_state_input_backward_is_the_token_sum_through_p(n: int, tokens: int, d: int, s: int):
    rng = np.random.default_rng(n * tokens + d + s)
    delta, C = rng.uniform(-1, 1, (n, tokens * d)), rng.uniform(-1, 1, (n, s))
    P, grad_P = rng.uniform(-1, 1, (d, s)), rng.uniform(-1, 1, (d, s))
    state_delta, new_grad_P = state_input_backward(_array(delta), _array(C), _array(P), _array(grad_P), tokens)
    sums = _array(_fold(delta.reshape(n, tokens, d), axis=1))
    assert _bits(_numpy(state_delta)) == _bits(_numpy(layer_downstream_batch(_array(P), sums)))
    expected = linear_accumulate_gradient_batch(sums, _array(C), _array(grad_P))
    assert _bits(_numpy(new_grad_P)) == _bits(_numpy(expected))


def test_the_large_state_input_and_gru_cases_are_threaded():
    n, _, d, s = STATE_INPUTS[-1]
    assert matmul_threads_for(n, s, d) > 1
    n, d, s = GRUS[-1]
    assert matmul_threads_for(n, d, 3 * s) > 1


def _pooling_forward_reference(X: np.ndarray, q: np.ndarray, tokens: int) -> tuple[np.ndarray, np.ndarray]:
    n, d = X.shape[0], q.shape[0]
    scores = _numpy(_array(X.reshape(n * tokens, d)) @ _array(q)).reshape(n, tokens)
    p = _numpy(array_softmax(_array(scores))).reshape(n, tokens)
    u = np.array([_numpy(_array(p[i : i + 1]) @ _array(X[i].reshape(tokens, d)))[0] for i in range(n)]).reshape(n, d)
    return u, p


@pytest.mark.parametrize("n,tokens,d", POOLINGS)
def test_attention_pooling_forward_is_its_composition(n: int, tokens: int, d: int):
    rng = np.random.default_rng(n + 10 * tokens + d)
    X, q = rng.uniform(-1, 1, (n, tokens * d)), rng.uniform(-1, 1, d)
    u, p = attention_pooling_forward(_array(X), _array(q), tokens)
    expected_u, expected_p = _pooling_forward_reference(X, q, tokens)
    assert _bits(_numpy(p)) == _bits(expected_p)
    assert _bits(_numpy(u)) == _bits(expected_u)


def test_a_zero_query_pools_the_mean_weights():
    X = np.random.default_rng(1).uniform(-1, 1, (3, 4 * 5))
    _, p = attention_pooling_forward(_array(X), Array.zeros(5), 4)
    assert _bits(_numpy(p)) == _bits(np.full((3, 4), 0.25))


@pytest.mark.parametrize("n,tokens,d", POOLINGS)
def test_attention_pooling_backward_is_its_composition(n: int, tokens: int, d: int):
    rng = np.random.default_rng(n * tokens + d)
    X, q = rng.uniform(-1, 1, (n, tokens * d)), rng.uniform(-1, 1, d)
    dU, grad_q = rng.uniform(-1, 1, (n, d)), rng.uniform(-1, 1, d)
    _, p = _pooling_forward_reference(X, q, tokens)
    dX, new_grad_q = attention_pooling_backward(_array(X), _array(p), _array(q), _array(dU), _array(grad_q), tokens)

    X3 = X.reshape(n, tokens, d)
    dp = np.array([_numpy(_array(X3[i]) @ _array(dU[i])) for i in range(n)]).reshape(n, tokens)
    dS = p * (dp - _fold(dp * p, axis=1)[:, np.newaxis])
    expected_grad_q = grad_q + _fold(dS.reshape(-1)[:, np.newaxis] * X3.reshape(-1, d), axis=0)
    expected_dX = p[:, :, np.newaxis] * dU[:, np.newaxis, :] + dS[:, :, np.newaxis] * q
    assert _bits(_numpy(new_grad_q)) == _bits(expected_grad_q)
    assert _bits(_numpy(dX)) == _bits(expected_dX.reshape(n, -1))


def _gru_inputs(n: int, d: int, s: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "u": rng.uniform(-1, 1, (n, d)),
        "c": rng.uniform(-1, 1, (n, s)),
        "W": rng.uniform(-1, 1, (3 * s, d)),
        "U": rng.uniform(-1, 1, (3 * s, s)),
        "b": rng.uniform(-1, 1, 3 * s),
    }


def _gru_forward_reference(g: dict[str, np.ndarray], s: int) -> tuple[np.ndarray, ...]:
    u, c, W, U, b = g["u"], g["c"], g["W"], g["U"], g["b"]
    A = _numpy(affine_forward_batch(_array(W), _array(u), _array(b)))
    rz = A[:, : 2 * s] + _numpy(linear_forward_batch(_array(U[: 2 * s]), _array(c)))
    r, z = _sigmoid(rz[:, :s]), _sigmoid(rz[:, s:])
    rc = r * c
    h_tilde = _tanh(A[:, 2 * s :] + _numpy(linear_forward_batch(_array(U[2 * s :]), _array(rc))))
    return z * c + (1.0 - z) * h_tilde, r, z, h_tilde, rc


@pytest.mark.parametrize("n,d,s", GRUS)
def test_gru_forward_is_its_composition(n: int, d: int, s: int):
    g = _gru_inputs(n, d, s, n + d + s)
    out = gru_forward(*(_array(g[k]) for k in ("u", "c", "W", "U", "b")))
    for actual, expected in zip(out, _gru_forward_reference(g, s), strict=True):
        assert _bits(_numpy(actual).reshape(n, s)) == _bits(expected)


def test_tanh_saturates_to_exactly_one_and_minus_one():
    # W = 0 and U = 0, so h~ = tanh(b_h); z = sigmoid(-inf) = 0, so h = h~
    s = 3
    b = np.array([0.0] * s + [-1e300] * s + [1e3, -1e3, 0.0])
    h, *_ = gru_forward(
        Array.zeros((1, 1)), Array.zeros((1, s)), Array.zeros((3 * s, 1)), Array.zeros((3 * s, s)), _array(b)
    )
    assert _bits(_numpy(h)) == _bits(np.array([[1.0, -1.0, 0.0]]))


@pytest.mark.parametrize("n,d,s", GRUS)
def test_gru_backward_is_its_composition(n: int, d: int, s: int):
    g = _gru_inputs(n, d, s, n * d + s)
    rng = np.random.default_rng(s)
    dH = rng.uniform(-1, 1, (n, s))
    grad_W, grad_U, grad_b = rng.uniform(-1, 1, (3 * s, d)), rng.uniform(-1, 1, (3 * s, s)), rng.uniform(-1, 1, 3 * s)
    _, r, z, h_tilde, rc = _gru_forward_reference(g, s)
    u, c, W, U = g["u"], g["c"], g["W"], g["U"]
    out = gru_backward(
        *(_array(v) for v in (dH, u, c, r, z, h_tilde, rc, W, U, grad_W, grad_U, grad_b)),
    )

    dA_h = dH * (1.0 - z) * (1.0 - h_tilde * h_tilde)
    d_rc = _numpy(layer_downstream_batch(_array(U[2 * s :]), _array(dA_h))).reshape(n, s)
    dA_r = d_rc * c * r * (1.0 - r)
    dA_z = dH * (c - h_tilde) * z * (1.0 - z)
    dA_rz = np.concatenate([dA_r, dA_z], axis=1)
    dA = np.concatenate([dA_rz, dA_h], axis=1)
    expected_W, expected_b = layer_accumulate_gradient_batch(_array(dA), _array(u), _array(grad_W), _array(grad_b))
    expected_U = np.concatenate(
        [
            _numpy(linear_accumulate_gradient_batch(_array(dA_rz), _array(c), _array(grad_U[: 2 * s]))),
            _numpy(linear_accumulate_gradient_batch(_array(dA_h), _array(rc), _array(grad_U[2 * s :]))),
        ]
    )
    expected_du = _numpy(layer_downstream_batch(_array(W), _array(dA)))
    dc_state = _numpy(layer_downstream_batch(_array(U[: 2 * s]), _array(dA_rz))).reshape(n, s)
    expected_dc = (dH * z + d_rc * r) + dc_state

    du, dc, new_W, new_U, new_b = out
    assert _bits(_numpy(du).reshape(n, d)) == _bits(expected_du)
    assert _bits(_numpy(dc).reshape(n, s)) == _bits(expected_dc)
    assert _bits(_numpy(new_W)) == _bits(_numpy(expected_W))
    assert _bits(_numpy(new_U)) == _bits(expected_U)
    assert _bits(_numpy(new_b)) == _bits(_numpy(expected_b))


def _codes_inputs(n: int, count: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(n + count + size)
    return rng.uniform(-1, 1, (n, size)), rng.uniform(-1, 1, (count, size))


def _nearest(H: np.ndarray, E: np.ndarray) -> np.ndarray:
    # CodesArrayLayer.nearest
    differences = H[:, np.newaxis, :] - E[np.newaxis, :, :]
    return np.argmin(_fold(differences * differences, axis=2), axis=1)


@pytest.mark.parametrize("n,count,size", CODES)
def test_codes_forward_snaps_each_row_to_numpys_nearest_code(n: int, count: int, size: int):
    H, E = _codes_inputs(n, count, size)
    Q, indices = codes_forward(_array(H), _array(E))
    expected = _nearest(H, E)
    assert indices == expected.tolist()
    assert _bits(_numpy(Q).reshape(n, size)) == _bits(E[expected])


def test_codes_forward_takes_the_lowest_index_on_a_tie_and_the_first_nan():
    E = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [np.nan, 0.0]])
    _, indices = codes_forward(_array(H), _array(E))
    assert indices == [0, 1, 0, 0] == _nearest(H, E).tolist()


@pytest.mark.parametrize("n,count,size", CODES)
def test_codes_backward_is_numpys_straight_through_and_scatter_add(n: int, count: int, size: int):
    H, E = _codes_inputs(n, count, size)
    rng = np.random.default_rng(count)
    dQ, grad_E = rng.uniform(-1, 1, (n, size)), rng.uniform(-1, 1, (count, size))
    weight, commitment = 1.0 / 3.0, 0.25
    indices = _nearest(H, E)
    Q = E[indices]
    dH, new_grad_E = codes_backward(
        _array(dQ), _array(H), _array(Q), indices.tolist(), _array(grad_E), weight, commitment
    )
    differences = H - Q
    expected_grad_E = grad_E.copy()
    np.add.at(expected_grad_E, indices, -2.0 * weight * differences)
    assert _bits(_numpy(new_grad_E)) == _bits(expected_grad_E)
    assert _bits(_numpy(dH).reshape(n, size)) == _bits(dQ + (2.0 * weight * commitment) * differences)


@pytest.mark.parametrize("n,count,size", CODES)
def test_codes_backward_holds_h_to_a_given_target_and_the_codes_to_h(n: int, count: int, size: int):
    # the commitment pulls h to the target's rows; the codebook's loss still pulls each code to its h
    H, E = _codes_inputs(n, count, size)
    rng = np.random.default_rng(count + 1)
    dQ, grad_E = rng.uniform(-1, 1, (n, size)), rng.uniform(-1, 1, (count, size))
    weight, commitment = 1.0 / 3.0, 0.25
    indices = _nearest(H, E)
    Q, T = E[indices], E[(indices + 1) % count]
    dH, new_grad_E = codes_backward(
        _array(dQ), _array(H), _array(Q), indices.tolist(), _array(grad_E), weight, commitment, _array(T)
    )
    expected_grad_E = grad_E.copy()
    np.add.at(expected_grad_E, indices, -2.0 * weight * (H - Q))
    assert _bits(_numpy(new_grad_E)) == _bits(expected_grad_E)
    assert _bits(_numpy(dH).reshape(n, size)) == _bits(dQ + (2.0 * weight * commitment) * (H - T))

    # q as the target is no target, by bits
    untargeted = codes_backward(_array(dQ), _array(H), _array(Q), indices.tolist(), _array(grad_E), weight, commitment)
    targeted = codes_backward(
        _array(dQ), _array(H), _array(Q), indices.tolist(), _array(grad_E), weight, commitment, _array(Q)
    )
    for a, b in zip(untargeted, targeted, strict=True):
        assert _bits(_numpy(a)) == _bits(_numpy(b))


def test_codes_backward_refuses_a_target_of_another_shape():
    with pytest.raises(ValueError, match="target"):
        codes_backward(
            Array.zeros((1, 2)),
            Array.zeros((1, 2)),
            Array.zeros((1, 2)),
            [0],
            Array.zeros((3, 2)),
            1.0,
            0.25,
            Array.zeros((2, 2)),
        )


def test_codes_backward_refuses_an_index_past_the_codes():
    with pytest.raises(ValueError, match="an index in"):
        codes_backward(
            Array.zeros((1, 2)), Array.zeros((1, 2)), Array.zeros((1, 2)), [3], Array.zeros((3, 2)), 1.0, 0.25
        )


@pytest.mark.parametrize("widths", [[1], [3, 5], [10, 10, 10, 1]])
def test_concatenate_columns_is_numpys(widths: list[int]):
    rng = np.random.default_rng(len(widths))
    arrays = [rng.uniform(-1, 1, (4, w)) for w in widths]
    out = concatenate_columns([_array(a) for a in arrays])
    assert _bits(_numpy(out)) == _bits(np.concatenate(arrays, axis=1))


def test_concatenate_columns_refuses_different_row_counts():
    with pytest.raises(ValueError, match="one row count"):
        concatenate_columns([Array.zeros((2, 3)), Array.zeros((3, 3))])
