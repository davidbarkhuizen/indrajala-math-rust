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
from array_helpers import bits, to_array, to_numpy

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


def _assert_bits(actual: Array, expected: np.ndarray) -> None:
    assert bits(to_numpy(actual).reshape(expected.shape)) == bits(expected)


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
    return [to_array(v) for v in values]


def _crate_codes(values: list[np.ndarray | None]) -> list[Array | None]:
    return [None if v is None else to_array(v) for v in values]


def _forward(
    X: np.ndarray, C: np.ndarray, params: Parameters, tokens: int, cross: bool, temperatures=None, couplings=None
) -> RecurrenceStep:
    return recurrence_step_forward(
        to_array(X),
        to_array(C),
        _crate(params.q),
        _crate(params.W),
        _crate(params.U),
        _crate(params.b),
        _crate_codes(params.E),
        tokens,
        cross=cross,
        temperatures=temperatures,
        couplings=couplings,
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
    straight=False,
):
    return recurrence_step_backward(
        step,
        to_array(dQ),
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
        straight=straight,
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
        u, p = attention_pooling_forward(to_array(X), to_array(params.q[g]), tokens)
        u = to_numpy(u).reshape(X.shape[0], -1)
        i = np.concatenate([u, _others(C, factors, g)], axis=1) if cross else u
        h, *gates = gru_forward(
            to_array(i), to_array(c), to_array(params.W[g]), to_array(params.U[g]), to_array(params.b[g])
        )
        E = params.E[g]
        temperature = None if temperatures is None else temperatures[g]
        weights = None
        if E is None:
            q, idx = h, None
        elif temperature is None:
            q, idx = codes_forward(h, to_array(E))
        else:
            q, weights, idx = soft_codes_forward(h, to_array(E), temperature)
        states.append(to_numpy(q).reshape(-1, s))
        hs.append(to_numpy(h).reshape(-1, s))
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
            E = to_array(params.E[g])
            dh, grad_e = soft_codes_backward(
                to_array(dq), h, q, weights, E, to_array(grads.E[g]), weight, commitments[g], temperature
            )
            dh = to_numpy(dh).reshape(n, s)
        else:
            target = None if targets is None or targets[g] is None else to_array(targets[g])
            dh, grad_e = codes_backward(to_array(dq), h, q, idx, to_array(grads.E[g]), weight, commitments[g], target)
            dh = to_numpy(dh).reshape(n, s)
        d_i, dc, gw, gu, gb = gru_backward(
            to_array(dh),
            to_array(i),
            to_array(c),
            r,
            z,
            h_tilde,
            rc,
            to_array(params.W[g]),
            to_array(params.U[g]),
            to_array(grads.W[g]),
            to_array(grads.U[g]),
            to_array(grads.b[g]),
        )
        d_i = to_numpy(d_i).reshape(n, -1)
        if cross:
            dC_cross.append(d_i[:, d:])
        du = d_i[:, :d]
        dx, gq = attention_pooling_backward(
            to_array(X), p, to_array(params.q[g]), to_array(du), to_array(grads.q[g]), tokens
        )
        dx = to_numpy(dx).reshape(n, -1)
        dX = dx if dX is None else dX + dx
        dC_own.append(to_numpy(dc).reshape(n, s))
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
            assert bits(to_numpy(actual)) == bits(to_numpy(value))
        actual_e = grad_codes[g]
        assert (actual_e is None) == (ge is None)
        if actual_e is not None and ge is not None:
            assert bits(to_numpy(actual_e)) == bits(to_numpy(ge))


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
    Q, indices = factored_codes_forward(to_array(H), sizes, _crate_codes(params.E))
    expected_Q, expected_indices = [], []
    for g, ((s, _), start) in enumerate(zip(factors, _starts(factors), strict=True)):
        h, E = H[:, start : start + s], params.E[g]
        if E is None:
            expected_Q.append(h)
            expected_indices.append(None)
        else:
            q, idx = codes_forward(to_array(h), to_array(E))
            expected_Q.append(to_numpy(q).reshape(n, s))
            expected_indices.append(idx)
    expected = np.concatenate(expected_Q, axis=1)
    _assert_bits(Q, expected)
    assert indices == expected_indices

    dQ = rng.uniform(-1, 1, H.shape)
    grads = [None if E is None else rng.uniform(-1, 1, E.shape) for E in params.E]
    weight, commitments = 0.5, [0.25] * len(factors)
    targets: list[np.ndarray | None] = _targets(params, indices) if targeted else [None] * len(factors)
    dH, grad_codes = factored_codes_backward(
        to_array(dQ), to_array(H), Q, sizes, indices, _crate_codes(grads), weight, commitments, _crate_codes(targets)
    )
    expected_dH = []
    for g, ((s, _), start) in enumerate(zip(factors, _starts(factors), strict=True)):
        dq, idx, grad = dQ[:, start : start + s], indices[g], grads[g]
        if idx is None or grad is None:
            expected_dH.append(dq)
            assert grad_codes[g] is None
            continue
        dh, ge = codes_backward(
            to_array(dq),
            to_array(H[:, start : start + s]),
            to_array(expected[:, start : start + s]),
            idx,
            to_array(grad),
            weight,
            commitments[g],
            None if (target := targets[g]) is None else to_array(target),
        )
        expected_dH.append(to_numpy(dh).reshape(n, s))
        actual = grad_codes[g]
        assert actual is not None and bits(to_numpy(actual)) == bits(to_numpy(ge))
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
        to_numpy(step.state).reshape(n, -1), _mirror(X, C, params, factors, tokens, cross), atol=1e-12
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
        np.testing.assert_allclose(to_numpy(actual).reshape(values.shape), central(values), rtol=1e-6, atol=1e-8)

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
            to_array(X),
            to_array(C),
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
        assert bits(to_numpy(a)) == bits(to_numpy(b))
    assert hard.indices == nones.indices


def test_the_step_refuses_a_temperature_for_a_continuous_factor_or_not_above_0():
    X, C, params = _inputs(2, 1, 3, [(2, 4), (2, None)], False, 1)
    with pytest.raises(ValueError, match="continuous factor's temperature"):
        _forward(X, C, params, 1, False, [None, 1.0])
    with pytest.raises(ValueError, match="temperature above 0"):
        _forward(X, C, params, 1, False, [0.0, None])
    with pytest.raises(ValueError, match="one entry per factor"):
        _forward(X, C, params, 1, False, [1.0])


@pytest.mark.parametrize("soft", [False, True])
def test_a_coupled_step_passes_on_its_blend_and_splits_its_gradient(soft: bool):
    # the state formation workplan's gradual handover: factor g's next state (1 - l) h + l q, its
    # snapped path's gradient l dq (the codes' losses whole), the cell's (1 - l) dq beside it
    factors: Factors = [(3, 5), (5, 7), (2, None)]
    couplings = [0.25, 0.7, 1.0]
    X, C, params = _inputs(6, 2, 9, factors, True, 41)
    temperatures = _temperatures(factors) if soft else None
    coupled = _forward(X, C, params, 2, True, temperatures, couplings)
    snapped = _forward(X, C, params, 2, True, temperatures)
    H, Q = to_numpy(snapped.h), to_numpy(snapped.state)
    starts = np.cumsum([0] + [s for s, _ in factors])
    want = Q.copy()
    for g, l in enumerate(couplings):
        block = slice(starts[g], starts[g + 1])
        if l < 1.0:
            want[:, block] = (1.0 - l) * H[:, block] + l * Q[:, block]
    _assert_bits(coupled.state, want)
    _assert_bits(coupled.h, H)
    assert coupled.indices == snapped.indices
    # the gradient: the snapped step's at l dq, plus the cell's own at (1 - l) dq (a continuous step)
    rng = np.random.default_rng(42)
    dQ = rng.uniform(-1, 1, C.shape)
    scale = np.concatenate([np.full(s, l) for (s, _), l in zip(factors, couplings, strict=True)])
    zeros = Parameters(
        [np.zeros_like(v) for v in params.q],
        [np.zeros_like(v) for v in params.W],
        [np.zeros_like(v) for v in params.U],
        [np.zeros_like(v) for v in params.b],
        [None if v is None else np.zeros_like(v) for v in params.E],
    )
    got = _backward(coupled, dQ, params, zeros, 0.5, [0.25, 0.25, 0.0])
    through = _backward(snapped, dQ * scale, params, zeros, 0.5, [0.25, 0.25, 0.0])
    free_params = params._replace(E=[None, None, None])
    free = _forward(X, C, free_params, 2, True)
    beside = _backward(free, dQ * (1.0 - scale), free_params, zeros._replace(E=[None, None, None]), 0.5, [0.0] * 3)

    def close(a: Array, b: Array, c: Array) -> None:
        np.testing.assert_allclose(to_numpy(a), to_numpy(b) + to_numpy(c), rtol=1e-12, atol=1e-12)

    for part in (0, 1):  # the tokens' gradient (one array: every factor read the same) and the state's
        a, b, c = got[part], through[part], beside[part]
        assert isinstance(a, Array) and isinstance(b, Array) and isinstance(c, Array)
        close(a, b, c)
    for parts in zip(got[2], through[2], beside[2], strict=True):
        close(*parts)
    for parts in zip(got[3], through[3], beside[3], strict=True):
        close(*parts)
    for parts in zip(got[4], through[4], beside[4], strict=True):
        close(*parts)
    for parts in zip(got[5], through[5], beside[5], strict=True):
        close(*parts)
    for a, b in zip(got[6], through[6], strict=True):  # the codes' gradients: the snapped path's alone
        assert (a is None) == (b is None)
        if a is not None and b is not None:
            np.testing.assert_allclose(to_numpy(a), to_numpy(b), rtol=1e-12, atol=1e-12)


def test_a_straight_soft_step_passes_dq_whole_to_its_cell_at_any_coupling():
    # the state formation workplan's D51: with no commitment, every part but the codes takes the
    # continuous step's gradient, dq whole; the codes take the soft step's as without straight
    factors: Factors = [(3, 5), (5, 7), (2, None)]
    couplings = [0.25, 0.9, 1.0]
    X, C, params = _inputs(6, 2, 9, factors, True, 44)
    temperatures = _temperatures(factors)
    coupled = _forward(X, C, params, 2, True, temperatures, couplings)
    dQ = np.random.default_rng(45).uniform(-1, 1, C.shape)
    zeros = Parameters(
        [np.zeros_like(v) for v in params.q],
        [np.zeros_like(v) for v in params.W],
        [np.zeros_like(v) for v in params.U],
        [np.zeros_like(v) for v in params.b],
        [None if v is None else np.zeros_like(v) for v in params.E],
    )
    got = _backward(coupled, dQ, params, zeros, 0.5, [0.0] * 3, straight=True)
    soft = _backward(coupled, dQ, params, zeros, 0.5, [0.0] * 3)
    free_params = params._replace(E=[None, None, None])
    free = _forward(X, C, free_params, 2, True)
    whole = _backward(free, dQ, free_params, zeros._replace(E=[None, None, None]), 0.5, [0.0] * 3)
    for part in (0, 1):
        a, b = got[part], whole[part]
        assert isinstance(a, Array) and isinstance(b, Array)
        np.testing.assert_allclose(to_numpy(a), to_numpy(b), rtol=1e-12, atol=1e-12)
    for a, b in [
        *zip(got[2], whole[2], strict=True),
        *zip(got[3], whole[3], strict=True),
        *zip(got[4], whole[4], strict=True),
        *zip(got[5], whole[5], strict=True),
    ]:
        np.testing.assert_allclose(to_numpy(a), to_numpy(b), rtol=1e-12, atol=1e-12)
    for a, b in zip(got[6], soft[6], strict=True):
        assert (a is None) == (b is None)
        if a is not None and b is not None:
            assert to_numpy(a).tolist() == to_numpy(b).tolist()


def test_couplings_of_1_are_the_snapped_step_by_bits():
    factors: Factors = [(3, 5), (5, 7), (2, None)]
    X, C, params = _inputs(6, 2, 9, factors, False, 43)
    plain, ones = _forward(X, C, params, 2, False), _forward(X, C, params, 2, False, None, [1.0, 1.0, 1.0])
    for a, b in [(plain.state, ones.state), (plain.h, ones.h)]:
        assert bits(to_numpy(a)) == bits(to_numpy(b))
    dQ = np.random.default_rng(44).uniform(-1, 1, C.shape)
    grads = _gradients(params, np.random.default_rng(45))
    want = _backward(plain, dQ, params, grads, 0.5, [0.25, 0.25, 0.0])
    got = _backward(ones, dQ, params, grads, 0.5, [0.25, 0.25, 0.0])
    _, dc, gq, gw, gu, gb, ge = got
    _, dc_want, gq_want, gw_want, gu_want, gb_want, ge_want = want
    arrays: list[tuple[Array | None, Array | None]] = [(dc, dc_want)]
    for mine, theirs in [(gq, gq_want), (gw, gw_want), (gu, gu_want), (gb, gb_want)]:
        arrays.extend(zip(mine, theirs, strict=True))
    arrays.extend(zip(ge, ge_want, strict=True))
    for a, b in arrays:
        assert (a is None) == (b is None)
        if a is not None and b is not None:
            assert bits(to_numpy(a)) == bits(to_numpy(b))


def test_the_step_refuses_a_coupling_out_of_range_or_below_1_for_a_continuous_factor():
    X, C, params = _inputs(2, 1, 3, [(2, 4), (2, None)], False, 1)
    with pytest.raises(ValueError, match="each coupling in \\[0, 1\\], factor 0's is 1.5"):
        _forward(X, C, params, 1, False, None, [1.5, 1.0])
    with pytest.raises(ValueError, match="each coupling in \\[0, 1\\], factor 0's is -0.1"):
        _forward(X, C, params, 1, False, None, [-0.1, 1.0])
    with pytest.raises(ValueError, match="couples only a factor with codes"):
        _forward(X, C, params, 1, False, None, [1.0, 0.5])
    with pytest.raises(ValueError, match="one entry per factor"):
        _forward(X, C, params, 1, False, None, [1.0])


def test_a_soft_factor_takes_no_commitment_target():
    factors: Factors = [(2, 4), (2, 4)]
    X, C, params = _inputs(2, 1, 3, factors, False, 1)
    step = _forward(X, C, params, 1, False, [1.0, None])
    grads = _gradients(params, np.random.default_rng(0))
    targets = _targets(params, step.indices)
    with pytest.raises(ValueError, match="soft factor's commitment target"):
        _backward(step, np.zeros(C.shape), params, grads, 1.0, [0.25, 0.25], targets)
