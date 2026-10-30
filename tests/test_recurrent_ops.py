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
from array_helpers import bits, to_array, to_numpy

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
    soft_codes_backward,
    soft_codes_forward,
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


def _fold(values: np.ndarray, axis: int) -> np.ndarray:
    return np.take(np.cumsum(values, axis=axis), -1, axis=axis)


def _crate_exp(values: np.ndarray) -> np.ndarray:
    return to_numpy(exp(to_array(values))).reshape(values.shape)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + _crate_exp(-values))


def _tanh(values: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 / (_crate_exp(2.0 * values) + 1.0)


@pytest.mark.parametrize("n,tokens,d,s", STATE_INPUTS)
def test_state_input_forward_adds_the_projected_state_to_every_token(n: int, tokens: int, d: int, s: int):
    rng = np.random.default_rng(n + tokens + d + s)
    X, C, P = rng.uniform(-1, 1, (n, tokens * d)), rng.uniform(-1, 1, (n, s)), rng.uniform(-1, 1, (d, s))
    out = to_numpy(state_input_forward(to_array(X), to_array(C), to_array(P), tokens))
    projected = to_numpy(linear_forward_batch(to_array(P), to_array(C)))
    expected = (X.reshape(n, tokens, d) + projected[:, np.newaxis, :]).reshape(n, -1)
    assert bits(out) == bits(expected)


@pytest.mark.parametrize("n,tokens,d,s", STATE_INPUTS)
def test_state_input_backward_is_the_token_sum_through_p(n: int, tokens: int, d: int, s: int):
    rng = np.random.default_rng(n * tokens + d + s)
    delta, C = rng.uniform(-1, 1, (n, tokens * d)), rng.uniform(-1, 1, (n, s))
    P, grad_P = rng.uniform(-1, 1, (d, s)), rng.uniform(-1, 1, (d, s))
    state_delta, new_grad_P = state_input_backward(to_array(delta), to_array(C), to_array(P), to_array(grad_P), tokens)
    sums = to_array(_fold(delta.reshape(n, tokens, d), axis=1))
    assert bits(to_numpy(state_delta)) == bits(to_numpy(layer_downstream_batch(to_array(P), sums)))
    expected = linear_accumulate_gradient_batch(sums, to_array(C), to_array(grad_P))
    assert bits(to_numpy(new_grad_P)) == bits(to_numpy(expected))


def test_the_large_state_input_and_gru_cases_are_threaded():
    n, _, d, s = STATE_INPUTS[-1]
    assert matmul_threads_for(n, s, d) > 1
    n, d, s = GRUS[-1]
    assert matmul_threads_for(n, d, 3 * s) > 1


def _pooling_forward_reference(X: np.ndarray, q: np.ndarray, tokens: int) -> tuple[np.ndarray, np.ndarray]:
    n, d = X.shape[0], q.shape[0]
    scores = to_numpy(to_array(X.reshape(n * tokens, d)) @ to_array(q)).reshape(n, tokens)
    p = to_numpy(array_softmax(to_array(scores))).reshape(n, tokens)
    u = np.array([to_numpy(to_array(p[i : i + 1]) @ to_array(X[i].reshape(tokens, d)))[0] for i in range(n)]).reshape(
        n, d
    )
    return u, p


@pytest.mark.parametrize("n,tokens,d", POOLINGS)
def test_attention_pooling_forward_is_its_composition(n: int, tokens: int, d: int):
    rng = np.random.default_rng(n + 10 * tokens + d)
    X, q = rng.uniform(-1, 1, (n, tokens * d)), rng.uniform(-1, 1, d)
    u, p = attention_pooling_forward(to_array(X), to_array(q), tokens)
    expected_u, expected_p = _pooling_forward_reference(X, q, tokens)
    assert bits(to_numpy(p)) == bits(expected_p)
    assert bits(to_numpy(u)) == bits(expected_u)


def test_a_zero_query_pools_the_mean_weights():
    X = np.random.default_rng(1).uniform(-1, 1, (3, 4 * 5))
    _, p = attention_pooling_forward(to_array(X), Array.zeros(5), 4)
    assert bits(to_numpy(p)) == bits(np.full((3, 4), 0.25))


@pytest.mark.parametrize("n,tokens,d", POOLINGS)
def test_attention_pooling_backward_is_its_composition(n: int, tokens: int, d: int):
    rng = np.random.default_rng(n * tokens + d)
    X, q = rng.uniform(-1, 1, (n, tokens * d)), rng.uniform(-1, 1, d)
    dU, grad_q = rng.uniform(-1, 1, (n, d)), rng.uniform(-1, 1, d)
    _, p = _pooling_forward_reference(X, q, tokens)
    dX, new_grad_q = attention_pooling_backward(
        to_array(X), to_array(p), to_array(q), to_array(dU), to_array(grad_q), tokens
    )

    X3 = X.reshape(n, tokens, d)
    dp = np.array([to_numpy(to_array(X3[i]) @ to_array(dU[i])) for i in range(n)]).reshape(n, tokens)
    dS = p * (dp - _fold(dp * p, axis=1)[:, np.newaxis])
    expected_grad_q = grad_q + _fold(dS.reshape(-1)[:, np.newaxis] * X3.reshape(-1, d), axis=0)
    expected_dX = p[:, :, np.newaxis] * dU[:, np.newaxis, :] + dS[:, :, np.newaxis] * q
    assert bits(to_numpy(new_grad_q)) == bits(expected_grad_q)
    assert bits(to_numpy(dX)) == bits(expected_dX.reshape(n, -1))


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
    A = to_numpy(affine_forward_batch(to_array(W), to_array(u), to_array(b)))
    rz = A[:, : 2 * s] + to_numpy(linear_forward_batch(to_array(U[: 2 * s]), to_array(c)))
    r, z = _sigmoid(rz[:, :s]), _sigmoid(rz[:, s:])
    rc = r * c
    h_tilde = _tanh(A[:, 2 * s :] + to_numpy(linear_forward_batch(to_array(U[2 * s :]), to_array(rc))))
    return z * c + (1.0 - z) * h_tilde, r, z, h_tilde, rc


@pytest.mark.parametrize("n,d,s", GRUS)
def test_gru_forward_is_its_composition(n: int, d: int, s: int):
    g = _gru_inputs(n, d, s, n + d + s)
    out = gru_forward(*(to_array(g[k]) for k in ("u", "c", "W", "U", "b")))
    for actual, expected in zip(out, _gru_forward_reference(g, s), strict=True):
        assert bits(to_numpy(actual).reshape(n, s)) == bits(expected)


def test_tanh_saturates_to_exactly_one_and_minus_one():
    # W = 0 and U = 0, so h~ = tanh(b_h); z = sigmoid(-inf) = 0, so h = h~
    s = 3
    b = np.array([0.0] * s + [-1e300] * s + [1e3, -1e3, 0.0])
    h, *_ = gru_forward(
        Array.zeros((1, 1)), Array.zeros((1, s)), Array.zeros((3 * s, 1)), Array.zeros((3 * s, s)), to_array(b)
    )
    assert bits(to_numpy(h)) == bits(np.array([[1.0, -1.0, 0.0]]))


@pytest.mark.parametrize("n,d,s", GRUS)
def test_gru_backward_is_its_composition(n: int, d: int, s: int):
    g = _gru_inputs(n, d, s, n * d + s)
    rng = np.random.default_rng(s)
    dH = rng.uniform(-1, 1, (n, s))
    grad_W, grad_U, grad_b = rng.uniform(-1, 1, (3 * s, d)), rng.uniform(-1, 1, (3 * s, s)), rng.uniform(-1, 1, 3 * s)
    _, r, z, h_tilde, rc = _gru_forward_reference(g, s)
    u, c, W, U = g["u"], g["c"], g["W"], g["U"]
    out = gru_backward(
        *(to_array(v) for v in (dH, u, c, r, z, h_tilde, rc, W, U, grad_W, grad_U, grad_b)),
    )

    dA_h = dH * (1.0 - z) * (1.0 - h_tilde * h_tilde)
    d_rc = to_numpy(layer_downstream_batch(to_array(U[2 * s :]), to_array(dA_h))).reshape(n, s)
    dA_r = d_rc * c * r * (1.0 - r)
    dA_z = dH * (c - h_tilde) * z * (1.0 - z)
    dA_rz = np.concatenate([dA_r, dA_z], axis=1)
    dA = np.concatenate([dA_rz, dA_h], axis=1)
    expected_W, expected_b = layer_accumulate_gradient_batch(
        to_array(dA), to_array(u), to_array(grad_W), to_array(grad_b)
    )
    expected_U = np.concatenate(
        [
            to_numpy(linear_accumulate_gradient_batch(to_array(dA_rz), to_array(c), to_array(grad_U[: 2 * s]))),
            to_numpy(linear_accumulate_gradient_batch(to_array(dA_h), to_array(rc), to_array(grad_U[2 * s :]))),
        ]
    )
    expected_du = to_numpy(layer_downstream_batch(to_array(W), to_array(dA)))
    dc_state = to_numpy(layer_downstream_batch(to_array(U[: 2 * s]), to_array(dA_rz))).reshape(n, s)
    expected_dc = (dH * z + d_rc * r) + dc_state

    du, dc, new_W, new_U, new_b = out
    assert bits(to_numpy(du).reshape(n, d)) == bits(expected_du)
    assert bits(to_numpy(dc).reshape(n, s)) == bits(expected_dc)
    assert bits(to_numpy(new_W)) == bits(to_numpy(expected_W))
    assert bits(to_numpy(new_U)) == bits(expected_U)
    assert bits(to_numpy(new_b)) == bits(to_numpy(expected_b))


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
    Q, indices = codes_forward(to_array(H), to_array(E))
    expected = _nearest(H, E)
    assert indices == expected.tolist()
    assert bits(to_numpy(Q).reshape(n, size)) == bits(E[expected])


def test_codes_forward_takes_the_lowest_index_on_a_tie_and_the_first_nan():
    E = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [np.nan, 0.0]])
    _, indices = codes_forward(to_array(H), to_array(E))
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
        to_array(dQ), to_array(H), to_array(Q), indices.tolist(), to_array(grad_E), weight, commitment
    )
    differences = H - Q
    expected_grad_E = grad_E.copy()
    np.add.at(expected_grad_E, indices, -2.0 * weight * differences)
    assert bits(to_numpy(new_grad_E)) == bits(expected_grad_E)
    assert bits(to_numpy(dH).reshape(n, size)) == bits(dQ + (2.0 * weight * commitment) * differences)


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
        to_array(dQ), to_array(H), to_array(Q), indices.tolist(), to_array(grad_E), weight, commitment, to_array(T)
    )
    expected_grad_E = grad_E.copy()
    np.add.at(expected_grad_E, indices, -2.0 * weight * (H - Q))
    assert bits(to_numpy(new_grad_E)) == bits(expected_grad_E)
    assert bits(to_numpy(dH).reshape(n, size)) == bits(dQ + (2.0 * weight * commitment) * (H - T))

    # q as the target is no target, by bits
    untargeted = codes_backward(
        to_array(dQ), to_array(H), to_array(Q), indices.tolist(), to_array(grad_E), weight, commitment
    )
    targeted = codes_backward(
        to_array(dQ), to_array(H), to_array(Q), indices.tolist(), to_array(grad_E), weight, commitment, to_array(Q)
    )
    for a, b in zip(untargeted, targeted, strict=True):
        assert bits(to_numpy(a)) == bits(to_numpy(b))


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


TEMPERATURES = [2.0, 0.25, 1.0 / 64.0]


def _soft(H: np.ndarray, E: np.ndarray, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    # the soft-to-hard quantiser's forward pass: (q, weights), the crate's exp
    differences = H[:, np.newaxis, :] - E[np.newaxis, :, :]
    D = _fold(differences * differences, axis=2)
    A = _crate_exp(-(D - D.min(axis=1)[:, np.newaxis]) / temperature)
    W = A / _fold(A, axis=1)[:, np.newaxis]
    return _fold(W[:, :, np.newaxis] * E[np.newaxis, :, :], axis=1), W


def _soft_backward(dQ, H, Q, W, E, grad_E, weight, commitment, temperature, straight=False):
    scale, code_scale = 2.0 / temperature, -2.0 * weight
    C = W * (_fold(dQ[:, np.newaxis, :] * E[np.newaxis, :, :], axis=2) - _fold(dQ * Q, axis=1)[:, np.newaxis])
    S = _fold(C[:, :, np.newaxis] * (E[np.newaxis, :, :] - Q[:, np.newaxis, :]), axis=1)
    dH = (dQ if straight else scale * S) + (2.0 * weight * commitment) * (H - Q)
    acc = grad_E.copy()
    for n in range(len(H)):  # row by row, each row's three terms in turn
        pulled = H[n][np.newaxis, :] - E
        acc = acc + W[n][:, np.newaxis] * dQ[n][np.newaxis, :]
        acc = acc + (scale * C[n])[:, np.newaxis] * pulled
        acc = acc + (code_scale * W[n])[:, np.newaxis] * pulled
    return dH, acc


@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("n,count,size", CODES)
def test_soft_codes_forward_is_numpys_softmax_mix(n: int, count: int, size: int, temperature: float):
    H, E = _codes_inputs(n, count, size)
    Q, W, indices = soft_codes_forward(to_array(H), to_array(E), temperature)
    expected_Q, expected_W = _soft(H, E, temperature)
    assert bits(to_numpy(Q).reshape(n, size)) == bits(expected_Q)
    assert bits(to_numpy(W).reshape(n, count)) == bits(expected_W)
    assert indices == _nearest(H, E).tolist()


def test_soft_codes_forward_nears_the_snap_as_it_cools():
    H, E = _codes_inputs(64, 16, 16)
    hot, _, _ = soft_codes_forward(to_array(H), to_array(E), 4.0)
    cold, W, indices = soft_codes_forward(to_array(H), to_array(E), 1e-4)
    snapped, _ = codes_forward(to_array(H), to_array(E))
    assert np.allclose(to_numpy(cold), to_numpy(snapped), atol=1e-9)
    assert not np.allclose(to_numpy(hot), to_numpy(snapped), atol=1e-3)
    assert np.allclose(to_numpy(W).reshape(64, 16).sum(axis=1), 1.0)
    assert (np.argmax(to_numpy(W).reshape(64, 16), axis=1) == indices).all()


@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("n,count,size", CODES)
def test_soft_codes_backward_is_numpys(n: int, count: int, size: int, temperature: float):
    H, E = _codes_inputs(n, count, size)
    rng = np.random.default_rng(count + 2)
    dQ, grad_E = rng.uniform(-1, 1, (n, size)), rng.uniform(-1, 1, (count, size))
    weight, commitment = 1.0 / 3.0, 0.25
    Q, W = _soft(H, E, temperature)
    dH, new_grad_E = soft_codes_backward(
        to_array(dQ),
        to_array(H),
        to_array(Q),
        to_array(W),
        to_array(E),
        to_array(grad_E),
        weight,
        commitment,
        temperature,
    )
    expected_dH, expected_grad_E = _soft_backward(dQ, H, Q, W, E, grad_E, weight, commitment, temperature)
    assert bits(to_numpy(dH).reshape(n, size)) == bits(expected_dH)
    assert bits(to_numpy(new_grad_E).reshape(count, size)) == bits(expected_grad_E)


@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("n,count,size", CODES)
def test_soft_codes_backward_straight_through_is_numpys(n: int, count: int, size: int, temperature: float):
    # the state formation workplan's D51: h's gradient dq plus the commitment's; e's as without it
    H, E = _codes_inputs(n, count, size)
    rng = np.random.default_rng(count + 3)
    dQ, grad_E = rng.uniform(-1, 1, (n, size)), rng.uniform(-1, 1, (count, size))
    weight, commitment = 1.0 / 3.0, 0.25
    Q, W = _soft(H, E, temperature)
    arrays = (to_array(dQ), to_array(H), to_array(Q), to_array(W), to_array(E), to_array(grad_E))
    dH, new_grad_E = soft_codes_backward(*arrays, weight, commitment, temperature, straight=True)
    expected_dH, expected_grad_E = _soft_backward(dQ, H, Q, W, E, grad_E, weight, commitment, temperature, True)
    assert bits(to_numpy(dH).reshape(n, size)) == bits(expected_dH)
    assert bits(to_numpy(new_grad_E).reshape(count, size)) == bits(expected_grad_E)
    _, soft_grad_E = soft_codes_backward(*arrays, weight, commitment, temperature)
    assert bits(to_numpy(new_grad_E)) == bits(to_numpy(soft_grad_E))


def test_soft_codes_backward_straight_through_nears_the_hard_snaps_as_it_cools():
    # cold, the soft step's straight-through gradient of h is the hard snap's (codes_backward); e's keeps
    # the soft step's task term w_k dq, which the hard snap's codes never take
    H, E = _codes_inputs(64, 16, 16)
    rng = np.random.default_rng(11)
    dQ, grad_E = rng.uniform(-1, 1, (64, 16)), np.zeros((16, 16))
    Q, W, _ = soft_codes_forward(to_array(H), to_array(E), 1e-4)
    dH, soft_grad_E = soft_codes_backward(
        to_array(dQ), to_array(H), Q, W, to_array(E), to_array(grad_E), 0.5, 0.25, 1e-4, straight=True
    )
    snapped, indices = codes_forward(to_array(H), to_array(E))
    hard_dH, hard_grad_E = codes_backward(to_array(dQ), to_array(H), snapped, indices, to_array(grad_E), 0.5, 0.25)
    np.testing.assert_allclose(to_numpy(dH), to_numpy(hard_dH), atol=1e-9)
    assert soft_grad_E.shape == hard_grad_E.shape


def test_soft_codes_backward_is_the_losss_gradient_by_central_differences():
    # L = sum(G * q) + weight (commitment ||h - sg[q]||^2 + sum_k sg[w_k] ||sg[h] - e_k||^2)
    n, count, size, temperature, weight, commitment = 3, 4, 5, 0.7, 0.4, 0.25
    H, E = _codes_inputs(n, count, size)
    G = np.random.default_rng(9).uniform(-1, 1, (n, size))
    Q, W = _soft(H, E, temperature)

    def loss(H_: np.ndarray, E_: np.ndarray) -> float:
        Q_, _ = _soft(H_, E_, temperature)
        committed = commitment * np.sum((H_ - Q) ** 2)  # q held fixed (sg)
        codebook = np.sum(W[:, :, np.newaxis] * (H[:, np.newaxis, :] - E_[np.newaxis, :, :]) ** 2)  # w, h fixed
        return float(np.sum(G * Q_) + weight * (committed + codebook))

    dH, grad_E = soft_codes_backward(
        to_array(G),
        to_array(H),
        to_array(Q),
        to_array(W),
        to_array(E),
        to_array(np.zeros((count, size))),
        weight,
        commitment,
        temperature,
    )
    step = 1e-6
    for X, analytic, of in ((H, to_numpy(dH).reshape(n, size), "h"), (E, to_numpy(grad_E).reshape(count, size), "e")):
        numeric = np.zeros_like(X)
        for i in np.ndindex(X.shape):
            up, down = X.copy(), X.copy()
            up[i] += step
            down[i] -= step
            args = ((up, E), (down, E)) if of == "h" else ((H, up), (H, down))
            numeric[i] = (loss(*args[0]) - loss(*args[1])) / (2 * step)
        assert np.allclose(analytic, numeric, atol=1e-6), of


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
def test_soft_codes_refuse_a_temperature_not_above_0_and_finite(temperature: float):
    with pytest.raises(ValueError, match="temperature"):
        soft_codes_forward(Array.zeros((1, 2)), Array.zeros((3, 2)), temperature)
    with pytest.raises(ValueError, match="temperature"):
        soft_codes_backward(
            Array.zeros((1, 2)),
            Array.zeros((1, 2)),
            Array.zeros((1, 2)),
            Array.zeros((1, 3)),
            Array.zeros((3, 2)),
            Array.zeros((3, 2)),
            1.0,
            0.25,
            temperature,
        )


@pytest.mark.parametrize("widths", [[1], [3, 5], [10, 10, 10, 1]])
def test_concatenate_columns_is_numpys(widths: list[int]):
    rng = np.random.default_rng(len(widths))
    arrays = [rng.uniform(-1, 1, (4, w)) for w in widths]
    out = concatenate_columns([to_array(a) for a in arrays])
    assert bits(to_numpy(out)) == bits(np.concatenate(arrays, axis=1))


def test_concatenate_columns_refuses_different_row_counts():
    with pytest.raises(ValueError, match="one row count"):
        concatenate_columns([Array.zeros((2, 3)), Array.zeros((3, 3))])
