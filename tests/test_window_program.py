"""
The window program (indrajala-ml's window-program workplan, stage 2): `WindowProgram`,
`window_program_forward` and `argmax_token_rows`. A program is checked by bits against the
per-layer path, the crate's functions called layer by layer as indrajala-ml's layers call them
(`WindowRecurrentRustArrayNetwork.forward_windows`): each op kind alone in a window, the studies'
windows (the linear window, the attention window of the division family), a rotary attention, two
factors, a continuous factor, the cross-term, and a batch over the threading threshold. The kept
states, cells and code indices are each window's step's.
"""

import math
from typing import Any

import numpy as np
import pytest

from indrajala_math_rust import (
    Array,
    Generator,
    WindowProgram,
    affine_forward_batch,
    argmax_token_rows,
    array_relu_mask,
    attention_accumulate_gradient_batch,
    attention_downstream_batch,
    attention_forward_batch,
    concatenate_columns,
    default_rng,
    embedding_accumulate_gradient,
    embedding_forward,
    layer_accumulate_gradient_batch,
    layer_downstream_batch,
    layer_norm_accumulate_gradient_batch,
    layer_norm_downstream_batch,
    layer_norm_forward_batch,
    layer_relu_forward_batch,
    layer_softmax_forward_batch,
    matmul_threads_for,
    recurrence_step_backward,
    recurrence_step_forward,
    state_input_backward,
    state_input_forward,
    sum_axis0,
    token_dropout_downstream,
    token_dropout_forward,
    window_program_forward,
    window_program_learn,
)

# each factor's (state size, codes; None for a continuous factor)
type Factors = list[tuple[int, int | None]]
type Op = tuple[Any, ...]


def _bits(array: Array) -> list[bytes]:
    # by bits, so -0.0 and 0.0 differ
    return [v.tobytes() for v in np.ravel(np.array(array.tolist(), dtype=np.float64))]


def _assert_bits(actual: Array, expected: Array) -> None:
    assert actual.shape == expected.shape
    assert _bits(actual) == _bits(expected)


def _random(rng: np.random.Generator, *shape: int) -> Array:
    return Array(rng.normal(0.0, 0.5, shape).tolist())


def _table(tokens: int, key_size: int) -> tuple[Array, Array]:
    # indrajala-ml's rotary_table: math's cos and sin of t * theta_j, (T, d_k / 2) each
    angles = [[t * 10000.0 ** (-2.0 * j / key_size) for j in range(key_size // 2)] for t in range(tokens)]
    return Array([[math.cos(a) for a in row] for row in angles]), Array([[math.sin(a) for a in row] for row in angles])


class Window:
    """A window's ops and parameters, built a layer at a time, as indrajala-ml's specs build them."""

    def __init__(self, tokens: int, vocabulary: int, features: int, seed: int) -> None:
        self.tokens = tokens
        self.vocabulary = vocabulary
        self.features = features
        self.rng = np.random.default_rng(seed)
        self.ops: list[Op] = [("embedding",)]
        self.parameters: list[list[Array]] = [[_random(self.rng, vocabulary, features)]]

    def add(self, op: Op, *parameters: Array) -> Window:
        self.ops.append(op)
        self.parameters.append(list(parameters))
        return self

    def position(self) -> Window:
        return self.add(("position",), _random(self.rng, self.tokens, self.features))

    def state_input(self, state_size: int) -> Window:
        return self.add(("state_input", self.tokens), _random(self.rng, self.features, state_size))

    def layer_norm(self) -> Window:
        d = self.features
        return self.add(("layer_norm", 1e-5), Array((1.0 + self.rng.normal(0, 0.1, d)).tolist()), _random(self.rng, d))

    def dense(self, size: int, activation: str) -> Window:
        W, b = _random(self.rng, size, self.features), _random(self.rng, size)
        self.features = size
        return self.add(("token_dense", self.tokens, activation), W, b)

    def attention(self, heads: int, causal: bool, rotary: bool = False, dropout: float = 0.0) -> Window:
        d = self.features
        weights = [array for _ in range(4) for array in (_random(self.rng, d, d), _random(self.rng, d))]
        cos, sin = _table(self.tokens, d // heads) if rotary else (None, None)
        return self.add(("attention", heads, causal, dropout, cos, sin), *weights)

    def fork(self) -> Window:
        return self.add(("fork",))

    def join(self) -> Window:
        return self.add(("add",))

    def dropout(self, p: float) -> Window:
        return self.add(("dropout", p))

    def output(self, classes: int) -> Window:
        W, b = _random(self.rng, classes, self.features), _random(self.rng, classes)
        return self.add(("token_output", self.tokens), W, b)

    def flat(self) -> list[Array]:
        return [array for arrays in self.parameters for array in arrays]


class Recurrence:
    """Each factor's pooling query, cell and codes, and initial state, in the network's order."""

    def __init__(self, features: int, factors: Factors, cross: bool, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.sizes = [s for s, _ in factors]
        self.snapped = [m is not None for _, m in factors]
        self.cross = cross
        total = sum(self.sizes)
        self.q, self.W, self.U, self.b, self.E, self.c = [], [], [], [], [], []
        for s, m in factors:
            self.q.append(_random(rng, features))
            self.W.append(_random(rng, 3 * s, features + (total - s if cross else 0)))
            self.U.append(_random(rng, 3 * s, s))
            self.b.append(_random(rng, 3 * s))
            self.E.append(None if m is None else _random(rng, m, s))
            self.c.append(_random(rng, s))

    def flat(self) -> list[Array]:
        return [
            array
            for g in range(len(self.sizes))
            for array in (self.q[g], self.W[g], self.U[g], self.b[g], self.E[g], self.c[g])
            if array is not None
        ]


def _per_layer(window: Window, recurrence: Recurrence, reads: int, X: Array, C0: Array):
    # the per-layer path: indrajala-ml's forward_windows, its layers' crate calls one by one
    k = window.tokens
    windows = X.shape[1] // k
    C = C0
    outputs, states, cells, indices = [], [], [], []
    for w in range(windows):
        a = X[:, w * k : (w + 1) * k]
        forks: list[Array] = []
        read = None
        for i, (op, parameters) in enumerate(zip(window.ops, window.parameters, strict=True)):
            if i == reads:
                read = a
            kind = op[0]
            if kind == "embedding":
                a = embedding_forward(a, parameters[0])
            elif kind == "position":
                (P,) = parameters
                a = a + P.reshape(P.shape[0] * P.shape[1])
            elif kind == "state_input":
                a = state_input_forward(a, C, parameters[0], op[1])
            elif kind == "fork":
                forks.append(a)
            elif kind == "add":
                a = a + forks.pop()
            elif kind == "layer_norm":
                gamma, beta = parameters
                a = layer_norm_forward_batch(a, gamma, beta, op[1])[0]
            elif kind in ("token_dense", "token_output"):
                W, b = parameters
                forward = (
                    layer_softmax_forward_batch
                    if kind == "token_output"
                    else layer_relu_forward_batch
                    if op[2] == "relu"
                    else affine_forward_batch
                )
                n, tokens = a.shape[0], op[1]
                rows = forward(W, a.reshape((n * tokens, W.shape[1])), b)
                a = rows.reshape((n, tokens * W.shape[0]))
            elif kind == "attention":
                _, heads, causal, dropout, cos, sin = op
                table = {} if cos is None else {"cos": cos, "sin": sin}
                result = attention_forward_batch(
                    a,
                    *parameters,
                    heads=heads,
                    causal=causal,
                    dropout=dropout,
                    training=False,
                    rng=default_rng(),
                    **table,
                )
                a = result[0]
            else:
                assert kind == "dropout"
        assert read is not None
        step = recurrence_step_forward(
            read, C, recurrence.q, recurrence.W, recurrence.U, recurrence.b, recurrence.E, k, cross=recurrence.cross
        )
        outputs.append(a)
        states.append(step.state)
        cells.append(step.h)
        indices.append(step.indices)
        C = step.state
    return concatenate_columns(outputs), states, cells, indices


def _ids(rng: np.random.Generator, n: int, columns: int, vocabulary: int) -> Array:
    return Array(rng.integers(0, vocabulary, (n, columns)).astype(float).tolist())


def _check(window: Window, recurrence: Recurrence, reads: int, n: int, windows: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    X = _ids(rng, n, windows * window.tokens, window.vocabulary)
    C0 = _random(rng, n, sum(recurrence.sizes))
    program = WindowProgram(
        window.ops, window.tokens, reads, recurrence.sizes, recurrence.snapped, cross=recurrence.cross
    )
    parameters = [*window.flat(), *recurrence.flat()]
    assert program.parameter_count == len(parameters)
    outputs, states, cells, indices = window_program_forward(program, X, parameters, C0, states=True, codes=True)
    expected = _per_layer(window, recurrence, reads, X, C0)
    _assert_bits(outputs, expected[0])
    assert len(states) == len(cells) == len(indices) == windows
    for state, cell, expected_state, expected_cell in zip(states, cells, expected[1], expected[2], strict=True):
        _assert_bits(state, expected_state)
        _assert_bits(cell, expected_cell)
    assert indices == expected[3]
    # only the outputs, unless asked
    plain = window_program_forward(program, X, parameters, C0)
    _assert_bits(plain[0], expected[0])
    assert plain[1:] == ([], [], [])


def _linear(tokens: int, state_size: int, seed: int) -> Window:
    # E2b''s window: the state in, one residual dense block, the output
    w = Window(tokens, 27, 16, seed).position().state_input(state_size)
    w.fork().layer_norm().dense(16, "linear").join()
    return w.layer_norm().output(3)


def _family(tokens: int, state_size: int, seed: int, rotary: bool = False, dropout: float = 0.0) -> Window:
    # the division family's window: three rounds of the state in, attention and an MLP block
    w = Window(tokens, 12, 16, seed).position()
    for _ in range(3):
        w.state_input(state_size)
        w.fork().layer_norm().attention(2, True, rotary).join()
        w.fork().layer_norm().dense(32, "relu")
        if dropout:
            w.dropout(dropout)
        w.dense(16, "linear").join()
    return w.layer_norm().output(5)


@pytest.mark.parametrize("n, windows", [(1, 1), (4, 3), (7, 9)])
def test_the_linear_window_is_the_per_layer_path(n: int, windows: int):
    window = _linear(1, 16, seed=1)
    _check(window, Recurrence(16, [(16, 16)], False, seed=2), len(window.ops) - 1, n, windows)


@pytest.mark.parametrize("reads_output", [True, False])
def test_the_family_window_is_the_per_layer_path(reads_output: bool):
    window = _family(3, 32, seed=3)
    reads = len(window.ops) - 1 if reads_output else window.ops.index(("state_input", 3))
    _check(window, Recurrence(16, [(16, 16), (16, 16)], False, seed=4), reads, 5, 6)


def test_a_rotary_window_with_dropout_is_the_per_layer_path_in_inference():
    window = _family(4, 8, seed=5, rotary=True, dropout=0.25)
    _check(window, Recurrence(16, [(8, 6)], False, seed=6), len(window.ops) - 1, 3, 4)


@pytest.mark.parametrize(
    "factors, cross",
    [([(4, None)], False), ([(3, 5), (5, None)], False), ([(3, 5), (5, 7)], True), ([(2, 3), (4, None), (2, 2)], True)],
)
def test_factors_continuous_and_crossed_are_the_per_layer_path(factors: Factors, cross: bool):
    size = sum(s for s, _ in factors)
    window = _linear(2, size, seed=7)
    _check(window, Recurrence(16, factors, cross, seed=8), len(window.ops) - 1, 4, 5)


@pytest.mark.parametrize(
    "build",
    [
        lambda w: w.position(),
        lambda w: w.layer_norm(),
        lambda w: w.dense(8, "relu"),
        lambda w: w.dense(8, "linear"),
        lambda w: w.attention(2, False),
        lambda w: w.attention(1, True, rotary=True),
        lambda w: w.dropout(0.5),
        lambda w: w.fork().layer_norm().join(),
    ],
)
def test_each_op_alone_is_the_per_layer_path(build):
    window = build(Window(3, 9, 8, seed=9).state_input(4))
    window.output(3)
    _check(window, Recurrence(window.features, [(4, 4)], False, seed=10), len(window.ops) - 1, 3, 2)


def test_a_batch_over_the_threading_threshold_is_the_per_layer_path():
    # the linear window at 64 features and two factors of 32, as the factored ops' threaded case
    n, d, s = 2048, 64, 32
    assert matmul_threads_for(n, d, 3 * s) > 1 and matmul_threads_for(n, d, d) > 1
    window = Window(1, 27, d, seed=11).position().state_input(2 * s)
    window.fork().layer_norm().dense(d, "linear").join()
    window.layer_norm().output(3)
    _check(window, Recurrence(d, [(s, 16), (s, 16)], False, seed=12), len(window.ops) - 1, n, 2)


def test_the_program_refuses_a_wrong_parameter_count_and_partial_windows():
    window = _linear(1, 4, seed=13)
    recurrence = Recurrence(16, [(4, 4)], False, seed=14)
    program = WindowProgram(window.ops, 1, len(window.ops) - 1, [4], [True], cross=False)
    parameters = [*window.flat(), *recurrence.flat()]
    X, C0 = Array([[1.0, 2.0]]), Array([[0.0] * 4])
    with pytest.raises(ValueError, match=r"requires \d+ parameters"):
        window_program_forward(program, X, parameters[:-1], C0)
    with pytest.raises(ValueError, match="c0 of shape"):
        window_program_forward(program, X, parameters, Array([[0.0] * 3]))
    wide = WindowProgram(window.ops, 3, len(window.ops) - 1, [4], [True], cross=False)
    with pytest.raises(ValueError, match="whole windows of 3 columns"):
        window_program_forward(wide, X, parameters, C0)


@pytest.mark.parametrize(
    "ops, reads, message",
    [
        ([("embedding",), ("sideways",)], 0, "has no op"),
        ([("embedding", 3)], 0, "takes 0 settings"),
        ([("token_dense", 2, "tanh")], 0, "relu"),
        ([("embedding",)], 1, "reads among them"),
    ],
)
def test_the_program_refuses_unknown_and_malformed_ops(ops, reads, message):
    with pytest.raises(ValueError, match=message):
        WindowProgram(ops, 1, reads, [4], [True], cross=False)


def _argmax_by_python(rows: list[list[float]], classes: int) -> list[tuple[int, ...]]:
    # indrajala-ml's RustBackend.argmax_token_rows before the crate op
    return [
        tuple(token.index(max(token)) for token in (row[i : i + classes] for i in range(0, len(row), classes)))
        for row in rows
    ]


def test_argmax_token_rows_is_pythons_index_of_max_ties_and_nans_included():
    nan = float("nan")
    rows = [
        [0.1, 0.7, 0.2, 0.5, 0.5, 0.0],
        [nan, 1.0, 0.0, 1.0, nan, 2.0],
        [0.0, 0.0, 0.0, -1.0, -1.0, -0.5],
        [-0.0, 0.0, nan, 2.0, 3.0, 3.0],
    ]
    assert argmax_token_rows(Array(rows), 3) == _argmax_by_python(rows, 3)
    rng = np.random.default_rng(15)
    values = rng.integers(0, 4, (50, 24)).astype(float).tolist()
    assert argmax_token_rows(Array(values), 4) == _argmax_by_python(values, 4)
    assert argmax_token_rows(Array(values), 1) == _argmax_by_python(values, 1)
    with pytest.raises(ValueError, match="whole runs"):
        argmax_token_rows(Array(values), 5)


# The training step (the workplan's stage 4): `window_program_learn` against indrajala-ml's
# `_accumulate_gradients` composed of the crate's ops, its recomputed forward pass (W20) included:
# every window forward in training with the generator's state kept, then from the last window each
# window's forward pass again from its generator's state, and its backward pass, layer by layer.


def _zeros(like: Array) -> Array:
    return Array.zeros(like.shape[0] if len(like.shape) == 1 else (like.shape[0], like.shape[1]))


def _forward_training(window: Window, read_at: int, a: Array, C: Array, rng: Generator):
    # one window's per-layer forward pass in training: each layer's input and result and its cache
    inputs, results, caches = [], [], []
    forks: list[Array] = []
    read = None
    for i, (op, parameters) in enumerate(zip(window.ops, window.parameters, strict=True)):
        if i == read_at:
            read = a
        inputs.append(a)
        kind, cache = op[0], None
        if kind == "embedding":
            a = embedding_forward(a, parameters[0])
        elif kind == "position":
            (P,) = parameters
            a = a + P.reshape(P.shape[0] * P.shape[1])
        elif kind == "state_input":
            a = state_input_forward(a, C, parameters[0], op[1])
        elif kind == "fork":
            forks.append(a)
        elif kind == "add":
            a = a + forks.pop()
        elif kind == "layer_norm":
            gamma, beta = parameters
            a, xhat, std = layer_norm_forward_batch(a, gamma, beta, op[1])
            cache = (xhat, std)
        elif kind in ("token_dense", "token_output"):
            W, b = parameters
            forward = (
                layer_softmax_forward_batch
                if kind == "token_output"
                else layer_relu_forward_batch
                if op[2] == "relu"
                else affine_forward_batch
            )
            n, tokens = a.shape[0], op[1]
            a = forward(W, a.reshape((n * tokens, W.shape[1])), b).reshape((n, tokens * W.shape[0]))
        elif kind == "attention":
            _, heads, causal, dropout, cos, sin = op
            table = {} if cos is None else {"cos": cos, "sin": sin}
            a, q, k, v, p, h, mask = attention_forward_batch(
                a, *parameters, heads=heads, causal=causal, dropout=dropout, training=True, rng=rng, **table
            )
            cache = (q, k, v, p, h, mask)
        else:
            a, cache = token_dropout_forward(a, op[1], rng)
        results.append(a)
        caches.append(cache)
    assert read is not None
    return inputs, results, caches, read


def _learn_per_layer(window: Window, recurrence: Recurrence, read_at: int, X: Array, Y: Array, C0: Array, rng):
    k, ops, L = window.tokens, window.ops, len(window.ops)
    windows = X.shape[1] // k
    weight = 1.0 / windows
    width = Y.shape[1] // windows
    grads = [[_zeros(p) for p in ps] for ps in window.parameters]
    g_parts = {name: [_zeros(a) for a in getattr(recurrence, name)] for name in ("q", "W", "U", "b")}
    g_codes = [None if e is None else _zeros(e) for e in recurrence.E]
    commitments = [0.25 if e is not None else 0.0 for e in recurrence.E]
    states, generators, C = [], [], C0
    for w in range(windows):
        states.append(C)
        generators.append(rng.state)
        _, _, _, read = _forward_training(window, read_at, X[:, w * k : (w + 1) * k], C, rng)
        C = recurrence_step_forward(
            read, C, recurrence.q, recurrence.W, recurrence.U, recurrence.b, recurrence.E, k, cross=recurrence.cross
        ).state
    after = rng.state
    dC = Array.zeros((X.shape[0], sum(recurrence.sizes)))
    adds = {}
    open_forks = []
    for i, op in enumerate(ops):
        if op[0] == "fork":
            open_forks.append(i)
        elif op[0] == "add":
            adds[open_forks.pop()] = i
    for w in reversed(range(windows)):
        rng.state = generators[w]
        inputs, results, caches, read = _forward_training(window, read_at, X[:, w * k : (w + 1) * k], states[w], rng)
        step = recurrence_step_forward(
            read,
            states[w],
            recurrence.q,
            recurrence.W,
            recurrence.U,
            recurrence.b,
            recurrence.E,
            k,
            cross=recurrence.cross,
        )
        pooled, dC_w, gq, gW, gU, gb, gE = recurrence_step_backward(
            step,
            dC,
            recurrence.q,
            recurrence.W,
            recurrence.U,
            g_parts["q"],
            g_parts["W"],
            g_parts["U"],
            g_parts["b"],
            g_codes,
            weight,
            commitments,
        )
        g_parts.update(q=gq, W=gW, U=gU, b=gb)
        g_codes = gE
        deltas: list[Array | None] = [None] * L
        attention_back: dict[int, tuple[Array, Array, Array, Array]] = {}
        state_back: dict[int, tuple[Array, Array]] = {}
        deltas[-1] = ((results[-1] - Y[:, w * width : (w + 1) * width]) / ops[-1][1]) * weight

        def downstream(j: int, deltas=deltas, caches=caches, attention_back=attention_back) -> Array:
            kind, ps, delta = ops[j][0], window.parameters[j], deltas[j]
            if kind == "fork":
                return downstream(j + 1, deltas, caches, attention_back) + deltas[adds[j]]
            assert delta is not None
            if kind in ("position", "add", "state_input"):
                return delta
            if kind == "layer_norm":
                return layer_norm_downstream_batch(delta, ps[0], *caches[j])
            if kind in ("token_dense", "token_output"):
                W, n, tokens = ps[0], delta.shape[0], ops[j][1]
                rows = layer_downstream_batch(W, delta.reshape((n * tokens, W.shape[0])))
                return rows.reshape((n, tokens * W.shape[1]))
            if kind == "attention":
                return attention_back[j][0]
            assert kind == "dropout"
            return token_dropout_downstream(delta, caches[j], ops[j][1])

        for i in reversed(range(L - 1)):
            kind, ps = ops[i][0], window.parameters[i]
            if kind == "fork":
                continue
            d = downstream(i + 1)
            if i + 1 == read_at:
                d = d + pooled
            if kind == "token_dense" and ops[i][2] == "relu":
                d = array_relu_mask(d, results[i])
            elif kind == "attention":
                _, heads, _, dropout, cos, sin = ops[i]
                q, kk, v, p, _, mask = caches[i]
                table = {} if cos is None else {"cos": cos, "sin": sin}
                attention_back[i] = attention_downstream_batch(
                    d, ps[0], ps[2], ps[4], ps[6], q, kk, v, p, heads, mask=mask, dropout=dropout, **table
                )
            elif kind == "state_input":
                state_back[i] = state_input_backward(d, states[w], ps[0], grads[i][0], ops[i][1])
            deltas[i] = d
        for i, op in enumerate(ops):
            kind, x, delta = op[0], inputs[i], deltas[i]
            if delta is None:
                continue  # a fork's, never read
            if kind == "embedding":
                grads[i][0] = embedding_accumulate_gradient(delta, x, grads[i][0])
            elif kind == "position":
                rows, cols = grads[i][0].shape[0], grads[i][0].shape[1]
                grads[i][0] = grads[i][0] + sum_axis0(delta).reshape((rows, cols))
            elif kind == "state_input":
                grads[i][0] = state_back[i][1]
            elif kind == "layer_norm":
                grads[i] = list(layer_norm_accumulate_gradient_batch(delta, caches[i][0], *grads[i]))
            elif kind in ("token_dense", "token_output"):
                W, n, tokens = window.parameters[i][0], delta.shape[0], op[1]
                grads[i] = list(
                    layer_accumulate_gradient_batch(
                        delta.reshape((n * tokens, W.shape[0])), x.reshape((n * tokens, W.shape[1])), *grads[i]
                    )
                )
            elif kind == "attention":
                _, dq, dk, dv = attention_back[i]
                grads[i] = list(attention_accumulate_gradient_batch(delta, x, caches[i][4], dq, dk, dv, *grads[i]))
        dC = dC_w
        for i in sorted(state_back):
            dC = dC + state_back[i][0]
    rng.state = after
    parts = [
        array
        for g in range(len(recurrence.sizes))
        for array in (
            g_parts["q"][g],
            g_parts["W"][g],
            g_parts["U"][g],
            g_parts["b"][g],
            g_codes[g],
            _zeros(recurrence.c[g]),
        )
        if array is not None
    ]
    return [array for arrays in grads for array in arrays] + parts, dC


def _check_learn(window: Window, recurrence: Recurrence, n: int, windows: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    X = _ids(rng, n, windows * window.tokens, window.vocabulary)
    classes = window.parameters[-1][0].shape[0]
    labels = rng.integers(0, classes, (n, windows * window.tokens))
    Y = Array(np.eye(classes)[labels].reshape(n, -1).tolist())
    C0 = _random(rng, n, sum(recurrence.sizes))
    program = WindowProgram(
        window.ops, window.tokens, len(window.ops) - 1, recurrence.sizes, recurrence.snapped, cross=recurrence.cross
    )
    parameters = [*window.flat(), *recurrence.flat()]
    gradients = [_zeros(p) for p in parameters]
    commitments = [0.25 if e is not None else 0.0 for e in recurrence.E]
    rng_program, rng_per_layer = default_rng(seed), default_rng(seed)
    dC = window_program_learn(program, X, Y, parameters, gradients, C0, rng_program, 1.0 / windows, commitments)
    expected, expected_dC = _learn_per_layer(window, recurrence, len(window.ops) - 1, X, Y, C0, rng_per_layer)
    assert len(gradients) == len(expected)
    for actual, wanted in zip(gradients, expected, strict=True):
        _assert_bits(actual, wanted)
    _assert_bits(dC, expected_dC)
    assert rng_program.state == rng_per_layer.state


@pytest.mark.parametrize("windows", [1, 2, 5])
def test_the_linear_windows_training_step_is_the_per_layer_path(windows: int):
    _check_learn(_linear(1, 16, seed=1), Recurrence(16, [(16, 16)], False, seed=2), 4, windows)


def test_the_family_windows_training_step_with_rotary_attention_and_dropout_is_the_per_layer_path():
    window = _family(3, 16, seed=3, rotary=True, dropout=0.25)
    _check_learn(window, Recurrence(16, [(8, 6), (8, 6)], False, seed=4), 5, 4)


@pytest.mark.parametrize(
    "factors, cross", [([(4, None)], False), ([(3, 5), (5, 7)], True), ([(2, 3), (4, None), (2, 2)], True)]
)
def test_factored_continuous_and_crossed_training_steps_are_the_per_layer_path(factors: Factors, cross: bool):
    size = sum(s for s, _ in factors)
    _check_learn(_linear(2, size, seed=7), Recurrence(16, factors, cross, seed=8), 4, 3)


def test_a_threaded_training_step_is_the_per_layer_path():
    n, d, s = 2048, 64, 32
    window = Window(1, 27, d, seed=11).position().state_input(2 * s)
    window.fork().layer_norm().dense(d, "relu").join()
    window.layer_norm().output(3)
    _check_learn(window, Recurrence(d, [(s, 16), (s, 16)], False, seed=12), n, 2)


def test_a_training_step_refuses_past_its_memory_budget_and_keeps_its_gradients():
    window, recurrence = _linear(1, 4, seed=13), Recurrence(16, [(4, 4)], False, seed=14)
    program = WindowProgram(window.ops, 1, len(window.ops) - 1, [4], [True], cross=False)
    parameters = [*window.flat(), *recurrence.flat()]
    gradients = [_zeros(p) for p in parameters]
    X, Y = Array([[1.0, 2.0]]), Array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="memory_budget"):
        window_program_learn(
            program, X, Y, parameters, gradients, Array([[0.0] * 4]), default_rng(0), 0.5, [0.25], memory_budget=100
        )
    assert all(g.shape == p.shape for g, p in zip(gradients, parameters, strict=True))
