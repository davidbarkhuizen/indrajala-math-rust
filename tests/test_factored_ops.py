"""
A factored window-recurrent state's ops (indrajala-ml's factored-state workplan, stage 2):
`factored_codes_*` and `recurrence_step_*`. Each is checked by bits against its composition from
the single-factor ops (`attention_pooling_*`, `gru_*`, `codes_*`), factor by factor, with the
shared sums in factor order (src/factored.rs): with one factor, that is today's window step by
bits (the workplan's D6). A numpy mirror, independent of the crate (numpy's own products and
`exp`), checks the step's values to within rounding and its gradients by central differences, the
cross-term's included. A case is over the threading threshold, so its threaded products are covered.
"""

from typing import NamedTuple

import numpy as np
import pytest

from indrajala_math_rust import (
    Array,
    RecurrenceStep,
    attention_pooling_backward,
    attention_pooling_forward,
    codes_backward,
    codes_forward,
    factored_codes_backward,
    factored_codes_forward,
    gru_backward,
    gru_forward,
    matmul_threads_for,
    recurrence_step_backward,
    recurrence_step_forward,
    soft_codes_backward,
    soft_codes_forward,
)

# each factor's (state size, codes; None for a continuous factor)
type Factors = list[tuple[int, int | None]]

# (examples, tokens, features, factors); the last over the threading threshold
STEPS: list[tuple[int, int, int, Factors]] = [
    (1, 1, 1, [(1, 1)]),
    (3, 4, 16, [(4, 8)]),
    (5, 1, 6, [(4, None)]),
    (3, 4, 16, [(4, 8), (4, 8)]),
    (6, 2, 9, [(3, 5), (5, 7), (2, None)]),
    (2048, 2, 64, [(32, 16), (32, 16)]),
]
# every step without the cross-term, and those of 2 or more factors with it
STEPS_AND_CROSS = [(*step, False) for step in STEPS] + [(*step, True) for step in STEPS if len(step[3]) > 1]
MIRRORED: list[tuple[Factors, bool]] = [
    ([(3, None)], False),
    ([(3, None), (2, None)], False),
    ([(3, None), (2, None)], True),
    ([(2, None), (4, None), (1, None)], False),
    ([(2, None), (4, None), (1, None)], True),
]


def _numpy(array: Array) -> np.ndarray:
    return np.array(array.tolist())


def _array(values: np.ndarray) -> Array:
    return Array(values.tolist())


def _bits(values: np.ndarray) -> list[bytes]:
    # by bits, so -0.0 and 0.0 differ, where np.array_equal takes them as equal
    return [v.tobytes() for v in np.ravel(values).astype(np.float64)]


def _assert_bits(actual: Array, expected: np.ndarray) -> None:
    assert _bits(_numpy(actual).reshape(expected.shape)) == _bits(expected)


class Parameters(NamedTuple):
    q: list[np.ndarray]
    W: list[np.ndarray]
    U: list[np.ndarray]
    b: list[np.ndarray]
    E: list[np.ndarray | None]


def _starts(factors: Factors) -> list[int]:
    return [sum(s for s, _ in factors[:g]) for g in range(len(factors))]


def _others(C: np.ndarray, factors: Factors, g: int) -> np.ndarray:
    start, size = _starts(factors)[g], factors[g][0]
    return np.concatenate([C[:, :start], C[:, start + size :]], axis=1)


def _inputs(n: int, tokens: int, d: int, factors: Factors, cross: bool, seed: int):
    rng = np.random.default_rng(seed)
    S = sum(s for s, _ in factors)
    X, C = rng.uniform(-1, 1, (n, tokens * d)), rng.uniform(-1, 1, (n, S))
    widths = [d + (S - s if cross else 0) for s, _ in factors]
    params = Parameters(
        [rng.uniform(-1, 1, d) for _ in factors],
        [rng.uniform(-1, 1, (3 * s, width)) for (s, _), width in zip(factors, widths, strict=True)],
        [rng.uniform(-1, 1, (3 * s, s)) for s, _ in factors],
        [rng.uniform(-1, 1, 3 * s) for s, _ in factors],
        [None if m is None else rng.uniform(-1, 1, (m, s)) for s, m in factors],
    )
    return X, C, params


def _crate(values: list[np.ndarray]) -> list[Array]:
    return [_array(v) for v in values]


def _crate_codes(values: list[np.ndarray | None]) -> list[Array | None]:
    return [None if v is None else _array(v) for v in values]


def _forward(
    X: np.ndarray, C: np.ndarray, params: Parameters, tokens: int, cross: bool, temperatures=None
) -> RecurrenceStep:
    return recurrence_step_forward(
        _array(X),
        _array(C),
        _crate(params.q),
        _crate(params.W),
        _crate(params.U),
        _crate(params.b),
        _crate_codes(params.E),
        tokens,
        cross=cross,
        temperatures=temperatures,
    )


def _temperatures(factors: Factors) -> list[float | None]:
    # each snapped factor soft, at a temperature of its own
    return [None if m is None else 0.5 + 0.25 * g for g, (_, m) in enumerate(factors)]


def _gradients(params: Parameters, rng: np.random.Generator) -> Parameters:
    return Parameters(
        *(
            [None if v is None else rng.uniform(-1, 1, v.shape) for v in values]  # type: ignore[union-attr]
            for values in params
        )
    )


def _targets(params: Parameters, indices: list[list[int] | None]) -> list[np.ndarray | None]:
    # each snapped factor's commitment target: the code after each row's own, so none is its q
    return [
        None if E is None or idx is None else E[(np.array(idx) + 1) % E.shape[0]]
        for E, idx in zip(params.E, indices, strict=True)
    ]


def _backward(
    step: RecurrenceStep,
    dQ: np.ndarray,
    params: Parameters,
    grads: Parameters,
    weight: float,
    commitments,
    targets=None,
):
    return recurrence_step_backward(
        step,
        _array(dQ),
        _crate(params.q),
        _crate(params.W),
        _crate(params.U),
        _crate(grads.q),  # type: ignore[arg-type]
        _crate(grads.W),  # type: ignore[arg-type]
        _crate(grads.U),  # type: ignore[arg-type]
        _crate(grads.b),  # type: ignore[arg-type]
        _crate_codes(grads.E),
        weight,
        commitments,
        None if targets is None else _crate_codes(targets),
    )


class Composed(NamedTuple):
    """The step from the single-factor ops, factor by factor."""

    state: np.ndarray
    h: np.ndarray
    indices: list[list[int] | None]
    saved: list[tuple]


def _compose_forward(
    X: np.ndarray, C: np.ndarray, params: Parameters, factors: Factors, tokens: int, cross: bool, temperatures=None
):
    states, hs, indices, saved = [], [], [], []
    for g, ((s, _), start) in enumerate(zip(factors, _starts(factors), strict=True)):
        c = C[:, start : start + s]
        u, p = attention_pooling_forward(_array(X), _array(params.q[g]), tokens)
        u = _numpy(u).reshape(X.shape[0], -1)
        i = np.concatenate([u, _others(C, factors, g)], axis=1) if cross else u
        h, *gates = gru_forward(_array(i), _array(c), _array(params.W[g]), _array(params.U[g]), _array(params.b[g]))
        E = params.E[g]
        temperature = None if temperatures is None else temperatures[g]
        weights = None
        if E is None:
            q, idx = h, None
        elif temperature is None:
            q, idx = codes_forward(h, _array(E))
        else:
            q, weights, idx = soft_codes_forward(h, _array(E), temperature)
        states.append(_numpy(q).reshape(-1, s))
        hs.append(_numpy(h).reshape(-1, s))
        indices.append(idx)
        saved.append((p, i, c, gates, h, q, idx, (temperature, weights)))
    return Composed(np.concatenate(states, axis=1), np.concatenate(hs, axis=1), indices, saved)


def _compose_backward(X, dQ, params, grads, composed, factors, tokens, cross, weight, commitments, targets=None):
    n = X.shape[0]
    S = sum(s for s, _ in factors)
    d = X.shape[1] // tokens
    dX, dC_own, dC_cross, new = None, [], [], []
    for g, ((s, _), start) in enumerate(zip(factors, _starts(factors), strict=True)):
        p, i, c, (r, z, h_tilde, rc), h, q, idx, (temperature, weights) = composed.saved[g]
        dq = dQ[:, start : start + s]
        if idx is None:
            dh, grad_e = dq, None
        elif weights is not None:
            E = _array(params.E[g])
            dh, grad_e = soft_codes_backward(
                _array(dq), h, q, weights, E, _array(grads.E[g]), weight, commitments[g], temperature
            )
            dh = _numpy(dh).reshape(n, s)
        else:
            target = None if targets is None or targets[g] is None else _array(targets[g])
            dh, grad_e = codes_backward(_array(dq), h, q, idx, _array(grads.E[g]), weight, commitments[g], target)
            dh = _numpy(dh).reshape(n, s)
        d_i, dc, gw, gu, gb = gru_backward(
            _array(dh),
            _array(i),
            _array(c),
            r,
            z,
            h_tilde,
            rc,
            _array(params.W[g]),
            _array(params.U[g]),
            _array(grads.W[g]),
            _array(grads.U[g]),
            _array(grads.b[g]),
        )
        d_i = _numpy(d_i).reshape(n, -1)
        if cross:
            dC_cross.append(d_i[:, d:])
        du = d_i[:, :d]
        dx, gq = attention_pooling_backward(_array(X), p, _array(params.q[g]), _array(du), _array(grads.q[g]), tokens)
        dx = _numpy(dx).reshape(n, -1)
        dX = dx if dX is None else dX + dx
        dC_own.append(_numpy(dc).reshape(n, s))
        new.append((gq, gw, gu, gb, grad_e))
    dC = np.concatenate(dC_own, axis=1)
    for g, piece in enumerate(dC_cross):
        start, size = _starts(factors)[g], factors[g][0]
        others = [j for j in range(S) if not start <= j < start + size]
        dC[:, others] += piece
    assert dX is not None
    return dX, dC, new


@pytest.mark.parametrize("targeted,soft", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("n,tokens,d,factors,cross", STEPS_AND_CROSS)
def test_the_step_is_its_composition_factor_by_factor(
    n: int, tokens: int, d: int, factors: Factors, cross: bool, targeted: bool, soft: bool
):
    # soft: each snapped factor's codes soft_codes_*, at its own temperature
    X, C, params = _inputs(n, tokens, d, factors, cross, n + tokens + d + len(factors))
    temperatures = _temperatures(factors) if soft else None
    step = _forward(X, C, params, tokens, cross, temperatures)
    composed = _compose_forward(X, C, params, factors, tokens, cross, temperatures)
    _assert_bits(step.state, composed.state)
    _assert_bits(step.h, composed.h)
    assert step.indices == composed.indices
    assert step.sizes == [s for s, _ in factors]

    rng = np.random.default_rng(n * d)
    dQ, grads = rng.uniform(-1, 1, composed.state.shape), _gradients(params, rng)
    weight, commitments = 1.0 / 3.0, [0.25 + 0.5 * g for g in range(len(factors))]
    targets = _targets(params, composed.indices) if targeted else None
    dX, dC, grad_q, grad_w, grad_u, grad_b, grad_codes = _backward(
        step, dQ, params, grads, weight, commitments, targets
    )
    expected_dX, expected_dC, expected = _compose_backward(
        X, dQ, params, grads, composed, factors, tokens, cross, weight, commitments, targets
    )
    assert isinstance(dX, Array)  # one array read: its poolings' gradients summed
    _assert_bits(dX, expected_dX)
    _assert_bits(dC, expected_dC)
    for g, (gq, gw, gu, gb, ge) in enumerate(expected):
        for actual, value in [(grad_q[g], gq), (grad_w[g], gw), (grad_u[g], gu), (grad_b[g], gb)]:
            assert _bits(_numpy(actual)) == _bits(_numpy(value))
        actual_e = grad_codes[g]
        assert (actual_e is None) == (ge is None)
        if actual_e is not None and ge is not None:
            assert _bits(_numpy(actual_e)) == _bits(_numpy(ge))


def test_the_large_step_is_threaded():
    n, _, d, factors = STEPS[-1]
    s = factors[0][0]
    assert matmul_threads_for(n, d, 3 * s) > 1


@pytest.mark.parametrize("targeted", [False, True])
@pytest.mark.parametrize("n,_tokens,_d,factors", STEPS)
def test_factored_codes_are_each_factors_codes(n: int, _tokens: int, _d: int, factors: Factors, targeted: bool):
    _, _, params = _inputs(n, 1, 1, factors, False, n + len(factors))
    rng = np.random.default_rng(n)
    sizes = [s for s, _ in factors]
    H = rng.uniform(-1, 1, (n, sum(sizes)))
    Q, indices = factored_codes_forward(_array(H), sizes, _crate_codes(params.E))
    expected_Q, expected_indices = [], []
    for g, ((s, _), start) in enumerate(zip(factors, _starts(factors), strict=True)):
        h, E = H[:, start : start + s], params.E[g]
        if E is None:
            expected_Q.append(h)
            expected_indices.append(None)
        else:
            q, idx = codes_forward(_array(h), _array(E))
            expected_Q.append(_numpy(q).reshape(n, s))
            expected_indices.append(idx)
    expected = np.concatenate(expected_Q, axis=1)
    _assert_bits(Q, expected)
    assert indices == expected_indices

    dQ = rng.uniform(-1, 1, H.shape)
    grads = [None if E is None else rng.uniform(-1, 1, E.shape) for E in params.E]
    weight, commitments = 0.5, [0.25] * len(factors)
    targets: list[np.ndarray | None] = _targets(params, indices) if targeted else [None] * len(factors)
    dH, grad_codes = factored_codes_backward(
        _array(dQ), _array(H), Q, sizes, indices, _crate_codes(grads), weight, commitments, _crate_codes(targets)
    )
    expected_dH = []
    for g, ((s, _), start) in enumerate(zip(factors, _starts(factors), strict=True)):
        dq, idx, grad = dQ[:, start : start + s], indices[g], grads[g]
        if idx is None or grad is None:
            expected_dH.append(dq)
            assert grad_codes[g] is None
            continue
        dh, ge = codes_backward(
            _array(dq),
            _array(H[:, start : start + s]),
            _array(expected[:, start : start + s]),
            idx,
            _array(grad),
            weight,
            commitments[g],
            None if (target := targets[g]) is None else _array(target),
        )
        expected_dH.append(_numpy(dh).reshape(n, s))
        actual = grad_codes[g]
        assert actual is not None and _bits(_numpy(actual)) == _bits(_numpy(ge))
    _assert_bits(dH, np.concatenate(expected_dH, axis=1))


# the numpy mirror: numpy's products and exp, nothing from the crate


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _mirror(X: np.ndarray, C: np.ndarray, params: Parameters, factors: Factors, tokens: int, cross: bool):
    """The step's next state before snapping, (N, S), for continuous factors: its smooth part."""
    n, d = X.shape[0], X.shape[1] // tokens
    X3 = X.reshape(n, tokens, d)
    hs = []
    for g, ((s, _), start) in enumerate(zip(factors, _starts(factors), strict=True)):
        scores = X3 @ params.q[g]
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        u = (p[:, :, np.newaxis] * X3).sum(axis=1)
        i = np.concatenate([u, _others(C, factors, g)], axis=1) if cross else u
        c = C[:, start : start + s]
        A = i @ params.W[g].T + params.b[g]
        U = params.U[g]
        r = _sigmoid(A[:, :s] + c @ U[:s].T)
        z = _sigmoid(A[:, s : 2 * s] + c @ U[s : 2 * s].T)
        h_tilde = np.tanh(A[:, 2 * s :] + (r * c) @ U[2 * s :].T)
        hs.append(z * c + (1.0 - z) * h_tilde)
    return np.concatenate(hs, axis=1)


@pytest.mark.parametrize("factors,cross", MIRRORED)
def test_the_step_agrees_with_the_numpy_mirror_and_its_gradients_with_central_differences(
    factors: Factors, cross: bool
):
    n, tokens, d = 3, 3, 4
    X, C, params = _inputs(n, tokens, d, factors, cross, 7 * len(factors))
    step = _forward(X, C, params, tokens, cross)
    np.testing.assert_allclose(
        _numpy(step.state).reshape(n, -1), _mirror(X, C, params, factors, tokens, cross), atol=1e-12
    )

    # the loss sum(R * state): its gradient is R, and the parameters' gradients start at zero
    rng = np.random.default_rng(len(factors))
    R = rng.uniform(-1, 1, C.shape)
    zeros = Parameters(*([None if v is None else np.zeros_like(v) for v in values] for values in params))  # type: ignore[arg-type]
    dX, dC, grad_q, grad_w, grad_u, grad_b, _ = _backward(step, R, params, zeros, 1.0, [0.25] * len(factors))

    def loss() -> float:
        return float(np.sum(R * _mirror(X, C, params, factors, tokens, cross)))

    def central(values: np.ndarray) -> np.ndarray:
        gradient, eps = np.zeros(values.shape), 1e-6
        for index in np.ndindex(values.shape):
            kept = values[index]
            values[index] = kept + eps
            up = loss()
            values[index] = kept - eps
            down = loss()
            values[index] = kept
            gradient[index] = (up - down) / (2 * eps)
        return gradient

    def close(actual: Array, values: np.ndarray) -> None:
        np.testing.assert_allclose(_numpy(actual).reshape(values.shape), central(values), rtol=1e-6, atol=1e-8)

    assert isinstance(dX, Array)
    close(dX, X)
    close(dC, C)
    for g in range(len(factors)):
        close(grad_q[g], params.q[g])
        close(grad_w[g], params.W[g])
        close(grad_u[g], params.U[g])
        close(grad_b[g], params.b[g])


def test_the_step_refuses_mismatched_factor_lists():
    X, C, params = _inputs(2, 1, 3, [(2, 4), (2, 4)], False, 1)
    with pytest.raises(ValueError, match="one entry per factor"):
        recurrence_step_forward(
            _array(X),
            _array(C),
            _crate(params.q),
            _crate(params.W),
            _crate(params.U),
            _crate(params.b[:1]),
            _crate_codes(params.E),
            1,
            cross=False,
        )


def test_the_cross_term_needs_two_factors():
    X, C, params = _inputs(2, 1, 3, [(2, 4)], False, 1)
    with pytest.raises(ValueError, match="2 or more factors for the cross-term"):
        _forward(X, C, params, 1, True)


def test_the_cross_term_needs_the_wider_input_weights():
    X, C, params = _inputs(2, 1, 3, [(2, 4), (2, 4)], False, 1)
    with pytest.raises(ValueError, match="each w of shape"):
        _forward(X, C, params, 1, True)


def test_a_continuous_factor_takes_no_commitment_target():
    H = Array.zeros((1, 2))
    with pytest.raises(ValueError, match="continuous factor's commitment target"):
        factored_codes_backward(H, H, H, [2], [None], [None], 0.5, [0.0], [Array.zeros((1, 2))])


def test_the_backward_pass_needs_a_codes_gradient_for_each_snapped_factor():
    X, C, params = _inputs(2, 1, 3, [(2, 4), (2, None)], False, 1)
    step = _forward(X, C, params, 1, False)
    grads = _gradients(params, np.random.default_rng(0))
    with pytest.raises(ValueError, match="a codes gradient for each snapped factor"):
        _backward(step, np.zeros(C.shape), params, grads._replace(E=[None, None]), 1.0, [0.25, 0.25])
    with pytest.raises(ValueError, match="None as a continuous factor's"):
        _backward(step, np.zeros(C.shape), params, grads._replace(E=[grads.E[0], np.zeros((4, 2))]), 1.0, [0.25] * 2)


def test_temperatures_of_none_are_the_hard_step_by_bits():
    factors: Factors = [(3, 5), (5, 7), (2, None)]
    X, C, params = _inputs(6, 2, 9, factors, False, 3)
    hard = _forward(X, C, params, 2, False)
    nones = _forward(X, C, params, 2, False, [None, None, None])
    for a, b in [(hard.state, nones.state), (hard.h, nones.h)]:
        assert _bits(_numpy(a)) == _bits(_numpy(b))
    assert hard.indices == nones.indices


def test_the_step_refuses_a_temperature_for_a_continuous_factor_or_not_above_0():
    X, C, params = _inputs(2, 1, 3, [(2, 4), (2, None)], False, 1)
    with pytest.raises(ValueError, match="continuous factor's temperature"):
        _forward(X, C, params, 1, False, [None, 1.0])
    with pytest.raises(ValueError, match="temperature above 0"):
        _forward(X, C, params, 1, False, [0.0, None])
    with pytest.raises(ValueError, match="one entry per factor"):
        _forward(X, C, params, 1, False, [1.0])


def test_a_soft_factor_takes_no_commitment_target():
    factors: Factors = [(2, 4), (2, 4)]
    X, C, params = _inputs(2, 1, 3, factors, False, 1)
    step = _forward(X, C, params, 1, False, [1.0, None])
    grads = _gradients(params, np.random.default_rng(0))
    targets = _targets(params, step.indices)
    with pytest.raises(ValueError, match="soft factor's commitment target"):
        _backward(step, np.zeros(C.shape), params, grads, 1.0, [0.25, 0.25], targets)
