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
    WindowProgram,
    affine_forward_batch,
    argmax_token_rows,
    attention_forward_batch,
    concatenate_columns,
    default_rng,
    embedding_forward,
    layer_norm_forward_batch,
    layer_relu_forward_batch,
    layer_softmax_forward_batch,
    matmul_threads_for,
    recurrence_step_forward,
    state_input_forward,
    window_program_forward,
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
