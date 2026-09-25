"""
The crate's RNG is numpy's legacy np.random (MT19937), bit for bit: after seed(s), random,
uniform, bernoulli_mask and the fused dropout forwards draw exactly what np.random.random,
np.random.uniform and np.random.random(shape) >= p draw after np.random.seed(s). The two
generators are separate states seeded alike. seed follows np.random.seed's three paths for every
seed: init_genrand for anything operator.index takes (after squeeze), init_by_array for other
sequences, OS entropy for None, and numpy's exception type for every seed numpy rejects.
"""

from __future__ import annotations

import array
import os
import subprocess
import sys
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import indrajala_math_rust as pa

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def as_numpy(arr: pa.Array) -> FloatArray:
    return np.array(arr.tolist(), dtype=np.float64)


def seed_both(seed: Any) -> None:
    np.random.seed(seed)
    pa.seed(seed)


def assert_identical(rust: pa.Array, reference: FloatArray) -> None:
    result = as_numpy(rust)
    assert result.shape == reference.shape
    # tobytes compares bits, so -0.0 and 0.0 differ and nothing hides behind a tolerance
    assert result.tobytes() == reference.tobytes()


# 312 doubles use one twist's 624 words, so 313 and 1000 cross twist boundaries
SHAPES: list[int | tuple[int, int]] = [1, 7, 312, 313, 1000, (1, 5), (3, 4), (40, 30), (32, 128)]


def numpy_shape(shape: int | tuple[int, int]) -> tuple[int, ...]:
    return (shape,) if isinstance(shape, int) else shape


@pytest.mark.parametrize("seed", [0, 1, 42, 12345, 2**31, 2**32 - 1])
@pytest.mark.parametrize("shape", SHAPES)
def test_random_and_uniform_match_numpy_after_the_same_seed(seed: int, shape: int | tuple[int, int]):
    seed_both(seed)
    assert_identical(pa.random(shape), np.random.random(numpy_shape(shape)))
    assert_identical(pa.uniform(-0.25, 0.75, shape), np.random.uniform(-0.25, 0.75, numpy_shape(shape)))


@pytest.mark.parametrize(("low", "high"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (5.0, 5.0), (-1e300, 1e300)])
def test_uniform_matches_numpy_at_any_finite_range(low: float, high: float):
    seed_both(7)
    assert_identical(pa.uniform(low, high, 500), np.random.uniform(low, high, 500))


@pytest.mark.parametrize(("low", "high"), [(0.0, float("inf")), (float("-inf"), 0.0), (float("nan"), 1.0)])
def test_uniform_raises_numpys_overflow_error_on_a_non_finite_range(low: float, high: float):
    with pytest.raises(OverflowError, match="Range exceeds valid bounds"):
        np.random.uniform(low, high, 3)
    with pytest.raises(OverflowError, match="Range exceeds valid bounds"):
        pa.uniform(low, high, 3)


@pytest.mark.parametrize("drop_probability", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("shape", [128, (32, 128), (512, 10)])
def test_bernoulli_mask_matches_numpys_dropout_mask(drop_probability: float, shape: int | tuple[int, int]):
    seed_both(3)
    reference = (np.random.random(numpy_shape(shape)) >= drop_probability).astype(np.float64)
    assert_identical(pa.bernoulli_mask(drop_probability, shape), reference)


def test_interleaved_calls_carry_the_position_across_calls_and_kinds():
    seed_both(2024)
    steps: list[tuple[Callable[[], pa.Array], Callable[[], FloatArray]]] = [
        (lambda: pa.uniform(-1.0, 1.0, (7, 11)), lambda: np.random.uniform(-1.0, 1.0, (7, 11))),
        (lambda: pa.uniform(0.0, 2.0, 5), lambda: np.random.uniform(0.0, 2.0, 5)),
        (lambda: pa.random(301), lambda: np.random.random(301)),
        (lambda: pa.bernoulli_mask(0.3, (9, 13)), lambda: (np.random.random((9, 13)) >= 0.3).astype(np.float64)),
        (lambda: pa.random((1, 1)), lambda: np.random.random((1, 1))),
        (lambda: pa.uniform(-3.0, 3.0, 640), lambda: np.random.uniform(-3.0, 3.0, 640)),
    ]
    for _ in range(3):
        for rust, reference in steps:
            assert_identical(rust(), reference())


def test_reseeding_restarts_the_stream():
    pa.seed(99)
    first = as_numpy(pa.random(700))
    pa.random(123)
    pa.seed(99)
    assert as_numpy(pa.random(700)).tobytes() == first.tobytes()


def dropout_inputs(batch: int | None) -> tuple[pa.Array, pa.Array, pa.Array]:
    weights = np.random.RandomState(5).uniform(-0.5, 0.5, (16, 12))
    inputs = np.random.RandomState(6).uniform(0.0, 1.0, (batch, 12) if batch else 12)
    bias = np.random.RandomState(7).uniform(-0.1, 0.1, 16)
    return pa.Array(weights.tolist()), pa.Array(inputs.tolist()), pa.Array(bias.tolist())


@pytest.mark.parametrize("batch", [None, 1, 32])
def test_fused_dropout_forward_draws_numpys_mask(batch: int | None):
    w, x, b = dropout_inputs(batch)
    seed_both(11)
    for _ in range(3):
        if batch is None:
            _a, mask, _base = pa.layer_dropout_forward(w, x, b, 0.4, True)
            reference = np.random.random(16) >= 0.4
        else:
            _a, mask, _base = pa.layer_dropout_forward_batch(w, x, b, 0.4, True)
            reference = np.random.random((batch, 16)) >= 0.4
        assert_identical(mask, reference.astype(np.float64))


@pytest.mark.parametrize("batch", [None, 32])
def test_dropout_forward_draws_nothing_when_not_training(batch: int | None):
    w, x, b = dropout_inputs(batch)
    seed_both(12)
    if batch is None:
        pa.layer_dropout_forward(w, x, b, 0.4, False)
    else:
        pa.layer_dropout_forward_batch(w, x, b, 0.4, False)
    assert_identical(pa.random(50), np.random.random(50))


# Seeds numpy accepts, by path; each must seed both generators identically.
INDEX_SEEDS: list[Any] = [
    5,
    True,
    False,
    np.int8(5),
    np.uint64(2**32 - 1),
    np.bool_(True),  # numpy warns it is deprecated as an index, but still takes it
    np.array(5),
    np.array([5]),
    np.array([[5]]),
    np.array([5], dtype=np.uint64),  # squeezes to a scalar, so the uint64 cast never happens
]
SEQUENCE_SEEDS: list[Any] = [
    [5],
    (5,),
    [0],
    [1, 2, 3],
    (2**32 - 1, 0, 7),
    range(10),
    list(range(1000)),  # longer than the 624-word state
    [True, 2],
    [True, False],
    [np.int8(1), 300],
    [np.uint32(1), 2],
    [np.array(5), 6],
    np.array([1, 2, 3]),
    np.array([1, 2, 3], dtype=np.int8),
    np.array([1, 2, 3], dtype=np.uint8),
    np.array([1, 2, 3], dtype=np.uint16),
    np.array([1, 2, 3], dtype=np.uint32),
    np.array([1, 2, 3], dtype=np.int32),
    np.array([True, False]),
    np.array([[1, 2, 3]]),  # squeezes to 1-D
    array.array("i", [5, 6]),
    array.array("B", [5, 6]),
    bytearray(b"\x05\x06"),
    memoryview(array.array("H", [5, 6])),
]


@pytest.mark.parametrize("seed", INDEX_SEEDS + SEQUENCE_SEEDS, ids=repr)
@pytest.mark.filterwarnings("ignore:In future, it will be an error for 'np.bool' scalars:DeprecationWarning")
def test_every_accepted_seed_gives_numpys_stream(seed: Any):
    seed_both(seed)
    assert_identical(pa.random(700), np.random.random(700))


def test_a_one_word_sequence_seeds_differently_from_the_int():
    pa.seed(5)
    from_int = as_numpy(pa.random(10))
    pa.seed([5])
    assert as_numpy(pa.random(10)).tobytes() != from_int.tobytes()


# Seeds numpy rejects, with the exception it raises; a message is checked word for word.
REJECTED_SEEDS: list[tuple[str, Any]] = [
    ("-1", -1),
    ("2**32", 2**32),
    ("2**70", 2**70),
    ("np.int64(-1)", np.int64(-1)),
    ("np.array(-1)", np.array(-1)),
    ("[]", []),
    ("()", ()),
    ("[[]]", [[]]),
    ("np.array([], float)", np.array([], dtype=np.float64)),
    ("np.zeros((0, 3))", np.zeros((0, 3))),
    ("[-1]", [-1]),
    ("[2**32]", [2**32]),
    ("[2**63 - 1]", [2**63 - 1]),
    ("[np.uint32(1), -1]", [np.uint32(1), -1]),
    ("[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
    ("[[-1]]", [[-1]]),
    ("np.array([[1, 2], [3, 4]])", np.array([[1, 2], [3, 4]])),
    ("np.array([True])", np.array([True])),  # squeezes to a 0-d bool, which isn't 1-D
    ("[1, [2]]", [1, [2]]),
    ("[[1], [2, 3]]", [[1], [2, 3]]),
    ("[[1.0], [2, 3]]", [[1.0], [2, 3]]),
    ("5.0", 5.0),
    ("1+0j", 1 + 0j),
    ("np.float64(5)", np.float64(5)),
    ("np.array(5.0)", np.array(5.0)),
    ("'5'", "5"),
    ("b'ab'", b"ab"),
    ("{}", {}),
    ("set()", set()),
    ("Fraction(5)", Fraction(5)),
    ("a generator", (x for x in [1])),
    ("[1.0]", [1.0]),
    ("[1, 2.0]", [1, 2.0]),
    ("['5']", ["5"]),
    ("[{}]", [{}]),
    ("[None]", [None]),
    ("[np.float32(1)]", [np.float32(1)]),
    ("[[1.0, 2.0]]", [[1.0, 2.0]]),
    ("[2**63]", [2**63]),
    ("[2**64]", [2**64]),
    ("[-2**63 - 1]", [-(2**63) - 1]),
    ("[np.int64(1), np.uint64(2)]", [np.int64(1), np.uint64(2)]),
    ("np.array([1, 2], uint64)", np.array([1, 2], dtype=np.uint64)),
    ("np.array([1.0, 2.0])", np.array([1.0, 2.0])),
    ("np.array([[1.0, 2.0], [3.0, 4.0]])", np.array([[1.0, 2.0], [3.0, 4.0]])),
    ("np.array([1, 2], object)", np.array([1, 2], dtype=object)),
    ("np.array([1, 2], timedelta64)", np.array([1, 2], dtype="m8[s]")),
    ("array.array('Q', [5])", array.array("Q", [5])),
    ("array.array('d', [5.0])", array.array("d", [5.0])),
]
NUMPY_MESSAGES = {"Seed must be between 0 and 2**32 - 1", "Seed must be non-empty", "Seed array must be 1-d"}


@pytest.mark.parametrize("seed", [seed for _label, seed in REJECTED_SEEDS], ids=[label for label, _ in REJECTED_SEEDS])
def test_every_rejected_seed_raises_numpys_exception_type(seed: Any):
    with pytest.raises((TypeError, ValueError)) as numpy_error:
        np.random.seed(seed)
    with pytest.raises((TypeError, ValueError)) as rust_error:
        pa.seed(seed)
    assert rust_error.type is numpy_error.type
    if str(numpy_error.value) in NUMPY_MESSAGES:
        assert str(rust_error.value) == str(numpy_error.value)


def test_a_rejected_seed_leaves_the_state_alone():
    seed_both(8)
    np.random.random(3)
    pa.random(3)
    for _label, seed in REJECTED_SEEDS:
        with pytest.raises((TypeError, ValueError)):
            pa.seed(seed)
    assert_identical(pa.random(20), np.random.random(20))


def test_seed_none_reseeds_from_entropy():
    pa.seed(1)
    first = as_numpy(pa.random(700))
    pa.seed(None)
    second = as_numpy(pa.random(700))
    pa.seed()
    third = as_numpy(pa.random(700))
    assert len({first.tobytes(), second.tobytes(), third.tobytes()}) == 3
    assert ((second >= 0.0) & (second < 1.0)).all()


DRAW = "import indrajala_math_rust as pa; {} print(pa.random(4).tolist())"


def draws_in_a_new_process(statement: str) -> str:
    return subprocess.run(
        [sys.executable, "-c", DRAW.format(statement)], capture_output=True, text=True, check=True
    ).stdout


@pytest.mark.parametrize("statement", ["", "pa.seed(None);"], ids=["unseeded", "seed(None)"])
def test_unseeded_draws_differ_between_processes(statement: str):
    assert draws_in_a_new_process(statement) != draws_in_a_new_process(statement)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_forked_children_inherit_the_state_as_numpys_global_does():
    pa.seed(None)
    read_end, write_end = os.pipe()
    pids = []
    for _ in range(2):
        pid = os.fork()
        if pid == 0:
            os.write(write_end, (repr(pa.random(4).tolist()) + "\n").encode())
            os._exit(0)
        pids.append(pid)
    for pid in pids:
        os.waitpid(pid, 0)
    os.close(write_end)
    with os.fdopen(read_end) as reader:
        children = reader.read().splitlines()
    assert len(children) == 2
    assert children[0] == children[1] == repr(pa.random(4).tolist())
