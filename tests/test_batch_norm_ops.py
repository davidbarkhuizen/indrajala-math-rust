"""
The batch-norm ops (`batch_norm_*`) and the bias-free linear ops before a dense one (`linear_*`),
against numpy transcriptions of indrajala-ml's README expressions (Batch normalization). After a
conv layer (`positions` > 1) the references work, as indrajala-ml's `BatchNormArrayLayer` does, on
the `(N * P, C)` rows view of the channel-major `(N, C * P)` batch, which the ops never build. The
batch-norm ops use only correctly rounded `+ - * /` and `sqrt` (and `exp` for the sigmoid), and
each sum over the batch is a left fold in row order, so they are compared by bits. `exp` isn't
correctly rounded: numpy's `np.exp` picks its implementation by CPU, and can differ from Rust's
`f64::exp` in the last bit, so the sigmoid references take the crate's `exp`. The linear ops
are the dense layer's products, `W @ x` and `grad_W + delta.T @ X`, without its bias.
"""

import numpy as np
import pytest

from indrajala_math_rust import (
    Array,
    batch_norm_accumulate_gradient_batch,
    batch_norm_downstream_batch,
    batch_norm_forward,
    batch_norm_forward_batch,
    exp,
    layer_accumulate_gradient_batch,
    linear_accumulate_gradient_batch,
    linear_forward,
    linear_forward_batch,
)

EPSILON = 1e-5
RATE = 0.1
ACTIVATIONS = ["sigmoid", "relu"]
# a batch of 2 is the smallest in training; 33 rows cross numpy's pairwise-summation block of 8
SHAPES = [(2, 1), (3, 2), (8, 5), (33, 7)]
# (examples, channels, positions) after a conv layer: one channel, a 1-position edge, and more
CONV_SHAPES = [(2, 1, 4), (3, 2, 9), (2, 3, 1), (5, 4, 16), (4, 6, 25)]
CASES = [(rows, cols, 1) for rows, cols in SHAPES] + CONV_SHAPES


def _numpy(array: Array) -> np.ndarray:
    return np.array(array.tolist())


def _bits(values: np.ndarray) -> list[bytes]:
    # by bits, so -0.0 and 0.0 differ, where np.array_equal takes them as equal
    return [v.tobytes() for v in np.ravel(values).astype(np.float64)]


def _fold(values: np.ndarray) -> np.ndarray:
    # each column's sum over the rows, a left fold from 0.0 in row order
    out = np.zeros(values.shape[1])
    for row in values:
        out = out + row
    return out


def _rows(values: np.ndarray, positions: int) -> np.ndarray:
    # channel-major (N, C * P) as (N * P, C), BatchNormArrayLayer._rows
    n = values.shape[0]
    return values.reshape(n, -1, positions).transpose(0, 2, 1).reshape(n * positions, -1)


def _flat(rows: np.ndarray, positions: int) -> np.ndarray:
    # _rows' inverse
    n = rows.shape[0] // positions
    return rows.reshape(n, positions, -1).transpose(0, 2, 1).reshape(n, -1)


def _activate(y: np.ndarray, activation: str) -> np.ndarray:
    # the crate's exp, Rust's f64::exp, which the ops call; np.exp can differ in the last bit
    return 1.0 / (1.0 + _numpy(exp(Array((-y).tolist())))) if activation == "sigmoid" else np.maximum(0.0, y)


def _case(rows: int, cols: int, seed: int, positions: int = 1) -> dict[str, np.ndarray]:
    # cols channels of positions values each per example
    rng = np.random.default_rng(seed)
    return {
        "x": rng.uniform(-3.0, 3.0, (rows, cols * positions)),
        "gamma": rng.uniform(0.5, 1.5, cols),
        "beta": rng.uniform(-0.5, 0.5, cols),
        "running_mean": rng.uniform(-1.0, 1.0, cols),
        "running_var": rng.uniform(0.5, 2.0, cols),
        "delta": rng.uniform(-1.0, 1.0, (rows, cols * positions)),
        "grad_gamma": rng.uniform(-1.0, 1.0, cols),
        "grad_beta": rng.uniform(-1.0, 1.0, cols),
    }


def _training_reference(c: dict[str, np.ndarray], activation: str, positions: int = 1) -> dict[str, np.ndarray]:
    x = _rows(c["x"], positions)
    m = x.shape[0]
    mu = _fold(x) / m
    d = x - mu
    ss = _fold(d * d)
    var = ss / m
    std = np.sqrt(var + EPSILON)
    xhat = d / std
    return {
        "a": _flat(_activate(c["gamma"] * xhat + c["beta"], activation), positions),
        "xhat": _flat(xhat, positions),
        "d": _flat(d, positions),
        "var": var,
        "std": std,
        "running_mean": (1 - RATE) * c["running_mean"] + RATE * mu,
        "running_var": (1 - RATE) * c["running_var"] + RATE * (ss / (m - 1)),
    }


def _downstream_reference(
    delta: np.ndarray, gamma: np.ndarray, t: dict[str, np.ndarray], positions: int = 1
) -> np.ndarray:
    d, var, std = _rows(t["d"], positions), t["var"], t["std"]
    m = d.shape[0]
    dxhat = _rows(delta, positions) * gamma
    inv_std = 1 / std
    inv_std3 = inv_std / (var + EPSILON)
    dvar = _fold(dxhat * d * -0.5 * inv_std3)
    dmu = _fold(dxhat * -inv_std) + dvar * _fold(-2 * d) / m
    return _flat(dxhat * inv_std + dvar * (2 * d) / m + dmu / m, positions)


def _training(
    c: dict[str, np.ndarray], activation: str, positions: int = 1, group_size: int | None = None
) -> dict[str, np.ndarray]:
    names = ["a", "xhat", "d", "var", "std", "running_mean", "running_var"]
    result = batch_norm_forward_batch(
        Array(c["x"].tolist()),
        Array(c["gamma"].tolist()),
        Array(c["beta"].tolist()),
        Array(c["running_mean"].tolist()),
        Array(c["running_var"].tolist()),
        EPSILON,
        RATE,
        activation,
        positions,
        group_size,
    )
    return {name: _numpy(array) for name, array in zip(names, result)}


def _group_ranges(rows: int, group_size: int) -> list[tuple[int, int]]:
    # ghost groups: runs of group_size examples in row order, the last one the remainder
    return [(first, min(first + group_size, rows)) for first in range(0, rows, group_size)]


def _ghost_reference(
    c: dict[str, np.ndarray], activation: str, positions: int, group_size: int
) -> dict[str, np.ndarray]:
    # each group is a batch of its own, the running averages carried from one to the next;
    # var and std are group-major
    running_mean, running_var = c["running_mean"], c["running_var"]
    parts: list[dict[str, np.ndarray]] = []
    for first, end in _group_ranges(c["x"].shape[0], group_size):
        group = {**c, "x": c["x"][first:end], "running_mean": running_mean, "running_var": running_var}
        part = _training_reference(group, activation, positions)
        running_mean, running_var = part["running_mean"], part["running_var"]
        parts.append(part)
    stacked = {name: np.concatenate([part[name] for part in parts]) for name in ("a", "xhat", "d", "var", "std")}
    return {**stacked, "running_mean": running_mean, "running_var": running_var}


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("rows, cols, positions", CASES)
@pytest.mark.parametrize("seed", range(5))
def test_the_training_forward_pass_is_the_readmes_by_bits(
    activation: str, rows: int, cols: int, positions: int, seed: int
):
    c = _case(rows, cols, seed, positions)
    expected = _training_reference(c, activation, positions)
    actual = _training(c, activation, positions)
    for name, values in expected.items():
        assert _bits(actual[name]) == _bits(values), name


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("rows, cols, positions", CASES)
def test_the_inference_forward_pass_is_the_readmes_by_bits_for_a_batch_and_each_row(
    activation: str, rows: int, cols: int, positions: int
):
    c = _case(rows, cols, 7, positions)
    xhat = (_rows(c["x"], positions) - c["running_mean"]) / np.sqrt(c["running_var"] + EPSILON)
    expected = _flat(_activate(c["gamma"] * xhat + c["beta"], activation), positions)
    gamma, beta, mean, var = (Array(c[name].tolist()) for name in ("gamma", "beta", "running_mean", "running_var"))

    batch = batch_norm_forward(Array(c["x"].tolist()), gamma, beta, mean, var, EPSILON, activation, positions)
    assert _bits(_numpy(batch)) == _bits(expected)
    for i in range(rows):
        row = batch_norm_forward(Array(c["x"][i].tolist()), gamma, beta, mean, var, EPSILON, activation, positions)
        assert row.shape == (cols * positions,)
        assert _bits(_numpy(row)) == _bits(expected[i])


@pytest.mark.parametrize("rows, cols, positions", CASES)
@pytest.mark.parametrize("seed", range(5))
def test_the_downstream_is_the_chain_rule_term_by_term_by_bits(rows: int, cols: int, positions: int, seed: int):
    c = _case(rows, cols, seed, positions)
    t = _training(c, "sigmoid", positions)
    expected = _downstream_reference(c["delta"], c["gamma"], t, positions)
    actual = batch_norm_downstream_batch(
        Array(c["delta"].tolist()),
        Array(c["gamma"].tolist()),
        Array(t["d"].tolist()),
        Array(t["var"].tolist()),
        Array(t["std"].tolist()),
        EPSILON,
        positions,
    )
    assert _bits(_numpy(actual)) == _bits(expected)


def test_the_downstream_matches_a_finite_difference_of_the_normalized_output():
    # the chain rule itself, not only its transcription: l = sum(delta * y), y = gamma * xhat + beta
    c = _case(5, 3, 11)
    t = _training(c, "sigmoid")
    dx = _downstream_reference(c["delta"], c["gamma"], t)

    def loss(x: np.ndarray) -> float:
        mu = x.mean(axis=0)
        xhat = (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=0) + EPSILON)
        return float(np.sum(c["delta"] * (c["gamma"] * xhat + c["beta"])))

    h = 1e-6
    for i in range(5):
        for j in range(3):
            up, down = c["x"].copy(), c["x"].copy()
            up[i, j] += h
            down[i, j] -= h
            assert dx[i, j] == pytest.approx((loss(up) - loss(down)) / (2 * h), rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("rows, cols, positions", CASES)
def test_accumulate_adds_the_folded_gamma_and_beta_gradients_by_bits(rows: int, cols: int, positions: int):
    c = _case(rows, cols, 3, positions)
    t = _training(c, "relu", positions)
    grad_gamma, grad_beta = batch_norm_accumulate_gradient_batch(
        Array(c["delta"].tolist()), Array(t["xhat"].tolist()), Array(c["grad_gamma"].tolist()),
        Array(c["grad_beta"].tolist()), positions,
    )  # fmt: skip
    delta, xhat = _rows(c["delta"], positions), _rows(t["xhat"], positions)
    assert _bits(_numpy(grad_gamma)) == _bits(c["grad_gamma"] + _fold(delta * xhat))
    assert _bits(_numpy(grad_beta)) == _bits(c["grad_beta"] + _fold(delta))


# (rows, cols, positions, group_size): groups that divide the batch, a remainder group, and
# conv groups
GHOST_CASES = [(4, 3, 1, 2), (8, 5, 1, 4), (7, 2, 1, 4), (34, 7, 1, 8), (6, 2, 4, 2), (5, 4, 9, 3), (8, 3, 1, 3)]


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("rows, cols, positions, group_size", GHOST_CASES)
def test_each_ghost_group_is_a_batch_of_its_own_by_bits(
    activation: str, rows: int, cols: int, positions: int, group_size: int
):
    c = _case(rows, cols, 13, positions)
    expected = _ghost_reference(c, activation, positions, group_size)
    actual = _training(c, activation, positions, group_size)
    for name, values in expected.items():
        assert _bits(actual[name]) == _bits(values), name


@pytest.mark.parametrize("rows, cols, positions", CASES)
@pytest.mark.parametrize("extra", [0, 1, 10])
def test_one_ghost_group_is_plain_batch_norm_by_bits(rows: int, cols: int, positions: int, extra: int):
    c = _case(rows, cols, 5, positions)
    plain = _training(c, "sigmoid", positions)
    grouped = _training(c, "sigmoid", positions, rows + extra)
    for name, values in plain.items():
        assert _bits(grouped[name]) == _bits(values), name


@pytest.mark.parametrize("rows, cols, positions, group_size", GHOST_CASES)
def test_the_ghost_downstream_is_each_groups_by_bits(rows: int, cols: int, positions: int, group_size: int):
    c = _case(rows, cols, 17, positions)
    t = _training(c, "relu", positions, group_size)
    parts = []
    for g, (first, end) in enumerate(_group_ranges(rows, group_size)):
        group = {
            "d": t["d"][first:end],
            "var": t["var"][g * cols : (g + 1) * cols],
            "std": t["std"][g * cols : (g + 1) * cols],
        }
        parts.append(_downstream_reference(c["delta"][first:end], c["gamma"], group, positions))
    actual = batch_norm_downstream_batch(
        Array(c["delta"].tolist()), Array(c["gamma"].tolist()), Array(t["d"].tolist()), Array(t["var"].tolist()),
        Array(t["std"].tolist()), EPSILON, positions, group_size,
    )  # fmt: skip
    assert _bits(_numpy(actual)) == _bits(np.concatenate(parts))


def test_ghost_groups_refuse_a_last_group_of_one():
    c = _case(5, 2, 0)
    with pytest.raises(ValueError, match="the last of 1"):
        _training(c, "sigmoid", 1, 2)
    t = _training(c, "sigmoid", 1, 3)
    with pytest.raises(ValueError, match="the last of 1"):
        batch_norm_downstream_batch(
            Array(c["delta"].tolist()), Array(c["gamma"].tolist()), Array(t["d"].tolist()), Array(t["var"].tolist()),
            Array(t["std"].tolist()), EPSILON, 1, 2,
        )  # fmt: skip


@pytest.mark.parametrize("group_size", [0, 1])
def test_a_group_size_under_2_is_refused(group_size: int):
    with pytest.raises(ValueError, match="group_size of 2 or more"):
        _training(_case(4, 2, 0), "relu", 1, group_size)


def test_the_downstream_refuses_statistics_for_other_groups():
    # one group's var and std for a batch of two groups
    c = _case(4, 2, 0)
    t = _training(c, "sigmoid")
    with pytest.raises(ValueError, match="var of shape"):
        batch_norm_downstream_batch(
            Array(c["delta"].tolist()), Array(c["gamma"].tolist()), Array(t["d"].tolist()), Array(t["var"].tolist()),
            Array(t["std"].tolist()), EPSILON, 1, 2,
        )  # fmt: skip


def test_a_column_of_negative_zeros_sums_to_positive_zero():
    # the fold starts from 0.0, and 0.0 + -0.0 is 0.0
    delta = Array([[-0.0, 1.0], [-0.0, 2.0]])
    zeros = Array.zeros(2)
    _grad_gamma, grad_beta = batch_norm_accumulate_gradient_batch(delta, Array([[1.0, 1.0], [1.0, 1.0]]), zeros, zeros)
    assert _bits(_numpy(grad_beta)) == _bits(np.array([0.0, 3.0]))


def test_training_refuses_a_batch_of_one():
    c = _case(1, 2, 0)
    with pytest.raises(ValueError, match="batch of 2 or more"):
        _training(c, "sigmoid")


def test_an_unknown_activation_is_refused():
    c = _case(2, 2, 0)
    with pytest.raises(ValueError, match="'sigmoid' or 'relu'"):
        _training(c, "tanh")


def test_mismatched_feature_counts_are_refused():
    x = Array([[1.0, 2.0], [3.0, 4.0]])
    two, three = Array.zeros(2), Array.zeros(3)
    with pytest.raises(ValueError, match="gamma"):
        batch_norm_forward(x, three, two, two, two, EPSILON, "relu")
    with pytest.raises(ValueError, match="2D batch of 3 channels x 1 positions"):
        batch_norm_forward_batch(x, three, three, three, three, EPSILON, RATE, "relu")
    with pytest.raises(ValueError, match="xhat"):
        batch_norm_accumulate_gradient_batch(x, Array([[1.0, 2.0]]), two, two)


def test_mismatched_positions_are_refused():
    # 2 values per example: 1 channel of 2 positions, not 1 channel of 3 or 2 channels of 2
    x, one, two = Array([[1.0, 2.0], [3.0, 4.0]]), Array.zeros(1), Array.zeros(2)
    with pytest.raises(ValueError, match="1 channels x 3 positions"):
        batch_norm_forward(x, one, one, one, one, EPSILON, "relu", 3)
    with pytest.raises(ValueError, match="2 channels x 2 positions"):
        batch_norm_forward_batch(x, two, two, two, two, EPSILON, RATE, "relu", 2)
    with pytest.raises(ValueError, match="2D batch of 1 channels x 3 positions"):
        batch_norm_downstream_batch(x, one, x, one, one, EPSILON, 3)
    with pytest.raises(ValueError, match="positions of 1 or more"):
        batch_norm_accumulate_gradient_batch(x, x, one, one, 0)
    batch_norm_forward_batch(x, one, one, one, Array([1.0]), EPSILON, RATE, "relu", 2)  # 1 channel of 2


@pytest.mark.parametrize("rows, size, input_size", [(1, 3, 4), (6, 5, 7), (40, 16, 33)])
def test_the_linear_ops_are_the_dense_ops_products_without_the_bias(rows: int, size: int, input_size: int):
    rng = np.random.default_rng(size)
    w = Array(rng.uniform(-1.0, 1.0, (size, input_size)).tolist())
    x_rows = rng.uniform(-1.0, 1.0, (rows, input_size))
    x = Array(x_rows.tolist())
    delta = Array(rng.uniform(-1.0, 1.0, (rows, size)).tolist())
    grad_w = Array(rng.uniform(-1.0, 1.0, (size, input_size)).tolist())

    # each row of the batch is the single-example product W @ x, as for layer_forward_batch
    batch = _numpy(linear_forward_batch(w, x))
    for i in range(rows):
        x_row = Array(x_rows[i].tolist())
        assert _bits(_numpy(linear_forward(w, x_row))) == _bits(_numpy(w @ x_row))
        assert _bits(batch[i]) == _bits(_numpy(w @ x_row))

    expected_grad_w, _grad_b = layer_accumulate_gradient_batch(delta, x, grad_w, Array.zeros(size))
    assert _bits(_numpy(linear_accumulate_gradient_batch(delta, x, grad_w))) == _bits(_numpy(expected_grad_w))


def test_linear_forward_refuses_a_batch():
    with pytest.raises(ValueError, match="1D x"):
        linear_forward(Array([[1.0, 2.0]]), Array([[1.0, 2.0]]))
