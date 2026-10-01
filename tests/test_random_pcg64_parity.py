"""
The crate's Generator is numpy's default_rng (Generator(PCG64(SeedSequence(seed)))), bit for bit:
pa.default_rng(s) draws exactly what np.random.default_rng(s) draws, through random, uniform,
bernoulli_mask and the fused dropout forwards given rng=. SeedSequence hashes and spawns as
numpy's does, and raises numpy's exception type for every seed numpy rejects. The state is
numpy's bit_generator.state dict, so it moves between numpy and the crate. The global MT19937
(test_random_numpy_parity.py) is a separate stream that a Generator never touches.
"""

from __future__ import annotations

import copy
import pickle
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import indrajala_math_rust as pa

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def as_numpy(arr: pa.Array) -> FloatArray:
    return np.array(arr.tolist(), dtype=np.float64)


def assert_identical(rust: pa.Array, reference: FloatArray) -> None:
    result = as_numpy(rust)
    assert result.shape == reference.shape
    # tobytes compares bits, so -0.0 and 0.0 differ and nothing hides behind a tolerance
    assert result.tobytes() == reference.tobytes()


def numpy_shape(shape: int | tuple[int, int]) -> tuple[int, ...]:
    return (shape,) if isinstance(shape, int) else shape


# Ints across the word boundaries SeedSequence splits them at, sequences, and numpy integers.
SEEDS: list[Any] = [
    0,
    1,
    7,
    42,
    2**32 - 1,
    2**32,
    2**64 + 1,
    2**127,
    2**128 + 3,
    2**200 + 5,
    [],
    [0],
    [1, 2, 3],
    (2**40, 1),
    range(5),
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    np.uint64(2**64 - 1),
    np.array([3, 4], dtype=np.int64),
    np.array([3, 4], dtype=np.uint32),
    ["0x10", "12", 3],
    [True, 2],
]
SEED_IDS = [repr(seed)[:40] for seed in SEEDS]
SHAPES: list[int | tuple[int, int]] = [1, 7, 1000, (1, 5), (3, 4), (40, 30), (32, 128)]


@pytest.mark.parametrize("seed", SEEDS, ids=SEED_IDS)
def test_seed_sequence_pool_and_state_match_numpy(seed: Any):
    reference = np.random.SeedSequence(seed)
    ours = pa.SeedSequence(seed)
    assert ours.pool == reference.pool.tolist()
    assert ours.generate_state(9) == reference.generate_state(9).tolist()
    assert ours.generate_state(5, np.uint64) == reference.generate_state(5, np.uint64).tolist()
    assert ours.generate_state(5, "uint64") == reference.generate_state(5, np.dtype("uint64")).tolist()
    assert ours.entropy is seed
    assert ours.spawn_key == () and ours.pool_size == 4 and ours.n_children_spawned == 0


@pytest.mark.parametrize("spawn_key", [(0,), (1, 2), (2**40,), ("0x5",), [3]])
def test_an_explicit_spawn_key_matches_numpy(spawn_key: Any):
    for entropy in [5, 2**100, [1, 2, 3, 4, 5]]:
        reference = np.random.SeedSequence(entropy, spawn_key=spawn_key)
        ours = pa.SeedSequence(entropy, spawn_key=spawn_key)
        assert ours.pool == reference.pool.tolist()
        assert ours.spawn_key == reference.spawn_key


def test_spawned_children_and_grandchildren_match_numpy():
    reference = np.random.SeedSequence(2024)
    ours = pa.SeedSequence(2024)
    # two spawns count on from each other, as numpy's do
    pairs = list(zip(reference.spawn(3) + reference.spawn(2), ours.spawn(3) + ours.spawn(2), strict=True))
    assert ours.n_children_spawned == reference.n_children_spawned == 5
    for numpy_child, our_child in pairs:
        assert our_child.spawn_key == numpy_child.spawn_key
        assert our_child.pool == numpy_child.pool.tolist()
        for numpy_grandchild, our_grandchild in zip(numpy_child.spawn(2), our_child.spawn(2), strict=True):
            assert our_grandchild.spawn_key == numpy_grandchild.spawn_key
            assert our_grandchild.pool == numpy_grandchild.pool.tolist()
    for numpy_child, our_child in pairs:
        numpy_rng = np.random.default_rng(numpy_child)
        assert_identical(pa.default_rng(our_child).random(64), numpy_rng.random(64))


@pytest.mark.parametrize("seed", SEEDS, ids=SEED_IDS)
def test_default_rng_starts_in_numpys_state(seed: Any):
    assert pa.default_rng(seed).state == np.random.default_rng(seed).bit_generator.state


@pytest.mark.parametrize("seed", [0, 1, 2**32, 2**64 + 1, 2**127, [5, 6]], ids=repr)
@pytest.mark.parametrize("shape", SHAPES)
def test_random_uniform_and_masks_match_numpy(seed: Any, shape: int | tuple[int, int]):
    ours = pa.default_rng(seed)
    reference = np.random.default_rng(seed)
    assert_identical(ours.random(shape), reference.random(numpy_shape(shape)))
    assert_identical(ours.uniform(-0.25, 0.75, shape), reference.uniform(-0.25, 0.75, numpy_shape(shape)))
    mask = (reference.random(numpy_shape(shape)) >= 0.3).astype(np.float64)
    assert_identical(ours.bernoulli_mask(0.3, shape), mask)


def test_seeds_whose_state_crosses_the_top_bit_match_numpy():
    # XSL-RR's rotation reads the state's top six bits, so states with the top bit set and clear
    # must both be covered over many steps
    top_bits = set()
    for seed in range(64):
        ours = pa.default_rng(seed)
        reference = np.random.default_rng(seed)
        for _ in range(4):
            top_bits.add(ours.state["state"]["state"] >> 127)
            assert_identical(ours.random(50), reference.random(50))
        assert ours.state == reference.bit_generator.state
    assert top_bits == {0, 1}


@pytest.mark.parametrize(("low", "high"), [(0.0, 1.0), (-1.0, 1.0), (5.0, 5.0), (-1e300, 1e300)])
def test_uniform_matches_numpy_at_any_finite_range(low: float, high: float):
    assert_identical(pa.default_rng(7).uniform(low, high, 500), np.random.default_rng(7).uniform(low, high, 500))


@pytest.mark.parametrize(
    ("low", "high"),
    [(0.0, float("inf")), (float("inf"), 0.0), (float("-inf"), 0.0), (float("nan"), 1.0), (-1e308, 1e308), (1.0, 0.0)],
)
def test_uniform_raises_numpys_errors_on_a_non_finite_or_negative_range(low: float, high: float):
    with pytest.raises((OverflowError, ValueError)) as numpy_error:
        np.random.default_rng(7).uniform(low, high, 3)
    with pytest.raises((OverflowError, ValueError)) as rust_error:
        pa.default_rng(7).uniform(low, high, 3)
    assert rust_error.type is numpy_error.type
    assert str(rust_error.value) == str(numpy_error.value)


@pytest.mark.parametrize("drop_probability", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_bernoulli_mask_matches_numpys_dropout_mask(drop_probability: float):
    reference = (np.random.default_rng(3).random((32, 128)) >= drop_probability).astype(np.float64)
    assert_identical(pa.default_rng(3).bernoulli_mask(drop_probability, (32, 128)), reference)


def test_interleaved_calls_carry_the_position_across_calls_and_kinds():
    ours = pa.default_rng(2024)
    reference = np.random.default_rng(2024)
    steps: list[tuple[Callable[[], pa.Array], Callable[[], FloatArray]]] = [
        (lambda: ours.uniform(-1.0, 1.0, (7, 11)), lambda: reference.uniform(-1.0, 1.0, (7, 11))),
        (lambda: ours.uniform(0.0, 2.0, 5), lambda: reference.uniform(0.0, 2.0, 5)),
        (lambda: ours.random(301), lambda: reference.random(301)),
        (lambda: ours.bernoulli_mask(0.3, (9, 13)), lambda: (reference.random((9, 13)) >= 0.3).astype(np.float64)),
        (lambda: ours.random((1, 1)), lambda: reference.random((1, 1))),
    ]
    for _ in range(3):
        for rust, numpy_draw in steps:
            assert_identical(rust(), numpy_draw())


def test_generators_are_independent_of_each_other_and_of_the_global_stream():
    pa.seed(5)
    np.random.seed(5)
    first, second = pa.default_rng(1), pa.default_rng(1)
    first.random(100)
    pa.random(17)
    assert_identical(second.random(10), np.random.default_rng(1).random(10))
    # neither generator moved the global MT19937
    pa.random(3)
    np.random.random(20)
    assert_identical(pa.random(5), np.random.random(5))


def dropout_inputs(batch: int | None) -> tuple[pa.Array, pa.Array, pa.Array]:
    weights = np.random.RandomState(5).uniform(-0.5, 0.5, (16, 12))
    inputs = np.random.RandomState(6).uniform(0.0, 1.0, (batch, 12) if batch else 12)
    bias = np.random.RandomState(7).uniform(-0.1, 0.1, 16)
    return pa.Array(weights.tolist()), pa.Array(inputs.tolist()), pa.Array(bias.tolist())


@pytest.mark.parametrize("batch", [None, 1, 32])
def test_fused_dropout_forward_with_rng_draws_numpys_generator_mask(batch: int | None):
    w, x, b = dropout_inputs(batch)
    ours = pa.default_rng(11)
    reference = np.random.default_rng(11)
    pa.seed(3)
    global_before = pa.random(4)
    pa.seed(3)
    for _ in range(3):
        if batch is None:
            _a, mask, _base = pa.layer_dropout_forward(w, x, b, 0.4, True, rng=ours)
            expected = reference.random(16) >= 0.4
        else:
            _a, mask, _base = pa.layer_dropout_forward_batch(w, x, b, 0.4, True, ours)
            expected = reference.random((batch, 16)) >= 0.4
        assert_identical(mask, expected.astype(np.float64))
    # the global stream didn't move
    assert_identical(pa.random(4), as_numpy(global_before))


@pytest.mark.parametrize("batch", [None, 32])
def test_dropout_forward_draws_nothing_from_rng_when_not_training(batch: int | None):
    w, x, b = dropout_inputs(batch)
    ours = pa.default_rng(12)
    if batch is None:
        pa.layer_dropout_forward(w, x, b, 0.4, False, rng=ours)
    else:
        pa.layer_dropout_forward_batch(w, x, b, 0.4, False, rng=ours)
    assert ours.state == np.random.default_rng(12).bit_generator.state


def test_a_state_moves_between_numpy_and_the_crate_and_continues_identically():
    reference = np.random.default_rng(99)
    reference.random(37)
    ours = pa.default_rng()
    ours.state = dict(reference.bit_generator.state)
    assert_identical(ours.random((5, 7)), reference.random((5, 7)))
    ours.uniform(-1.0, 1.0, 13)
    reference.bit_generator.state = ours.state
    assert_identical(ours.random(100), reference.random(100))
    assert ours.state == reference.bit_generator.state


def test_the_buffered_32_bit_fields_round_trip_unchanged():
    reference = np.random.default_rng(4)
    reference.integers(0, 2**32, dtype=np.uint32)  # buffers half a draw: has_uint32 = 1
    state = reference.bit_generator.state
    assert state["has_uint32"] == 1
    ours = pa.default_rng()
    ours.state = dict(state)
    assert ours.state == state
    assert_identical(ours.random(10), reference.random(10))


def test_reading_the_state_returns_a_copy():
    ours = pa.default_rng(1)
    state = ours.state
    state["state"]["state"] = 0
    assert ours.state == np.random.default_rng(1).bit_generator.state


# Invalid states, with what numpy's setter raises for each.
REJECTED_STATES: list[tuple[str, Any, type[Exception]]] = [
    ("not a dict", 5, TypeError),
    ("another generator", {"bit_generator": "MT19937"}, ValueError),
    ("no bit_generator", {"state": {"state": 1, "inc": 1}}, ValueError),
    ("no inc", {"bit_generator": "PCG64", "state": {"state": 1}}, KeyError),
    ("state over 128 bits", {"bit_generator": "PCG64", "state": {"state": 2**128, "inc": 1}}, OverflowError),
    ("negative inc", {"bit_generator": "PCG64", "state": {"state": 1, "inc": -1}}, OverflowError),
]


@pytest.mark.parametrize(
    ("state", "error"),
    [(state, error) for _label, state, error in REJECTED_STATES],
    ids=[label for label, _state, _error in REJECTED_STATES],
)
def test_a_rejected_state_raises_numpys_exception_type_and_changes_nothing(state: Any, error: type[Exception]):
    full = {"has_uint32": 0, "uinteger": 0} | state if isinstance(state, dict) else state
    reference = np.random.default_rng(6)
    ours = pa.default_rng(6)
    with pytest.raises(error):
        reference.bit_generator.state = full
    with pytest.raises(error):
        ours.state = full
    assert ours.state == reference.bit_generator.state


# Seeds numpy rejects, with its exception type and message.
REJECTED_SEEDS: list[tuple[str, Any]] = [
    ("negative int", -1),
    ("negative numpy int", np.int8(-3)),
    ("float", 1.5),
    ("str", "12"),
    ("bytes", b"12"),
    ("array.array", __import__("array").array("i", [1, 2])),
    ("float element", [1.0]),
    ("numpy float element", [np.float64(1.0)]),
    ("negative element", [-1]),
    ("float array", np.array([1.0, 2.0])),
    ("negative array", np.array([-1])),
    ("0-d array", np.array(5)),
    ("nested list", [[1, 2], [3, 4]]),
    ("nested tuple", [(1,)]),
    ("nested range", [range(2)]),
    ("nested array", [np.array([1, 2])]),
    ("2-d array", np.array([[1, 2], [3, 4]])),
    ("bytes element", [b"1"]),
    ("None element", [None]),
    ("numpy bool element", [np.bool_(True)]),
    ("unrecognized string", ["abc"]),
    ("negative hex string", ["-0x5"]),
    ("bad hex string", ["0x"]),
    ("octal string", ["0o7"]),
]


@pytest.mark.parametrize("seed", [seed for _label, seed in REJECTED_SEEDS], ids=[label for label, _ in REJECTED_SEEDS])
def test_every_rejected_seed_raises_numpys_exception_and_message(seed: Any):
    with pytest.raises((TypeError, ValueError)) as numpy_error:
        np.random.default_rng(seed)
    with pytest.raises((TypeError, ValueError)) as rust_error:
        pa.default_rng(seed)
    assert rust_error.type is numpy_error.type
    assert str(rust_error.value) == str(numpy_error.value)


@pytest.mark.parametrize("spawn_key", [(-1,), ((1,),), 5], ids=repr)
def test_a_rejected_spawn_key_raises_numpys_exception_type(spawn_key: Any):
    with pytest.raises(TypeError if spawn_key != (-1,) else ValueError) as numpy_error:
        np.random.SeedSequence(1, spawn_key=spawn_key)
    with pytest.raises(numpy_error.type):
        pa.SeedSequence(1, spawn_key=spawn_key)


@pytest.mark.parametrize("dtype", ["float64", np.int32, np.dtype("float32")], ids=repr)
def test_generate_state_rejects_other_dtypes_as_numpy_does(dtype: Any):
    with pytest.raises(ValueError, match="only support uint32 or uint64"):
        np.random.SeedSequence(1).generate_state(2, dtype)
    with pytest.raises(ValueError, match="only support uint32 or uint64"):
        pa.SeedSequence(1).generate_state(2, dtype)


def test_default_rng_returns_a_generator_as_it_is():
    rng = pa.default_rng(3)
    assert pa.default_rng(rng) is rng
    assert isinstance(pa.Generator(3), pa.Generator)
    assert pa.Generator(3).state == rng.state


def test_none_draws_fresh_entropy():
    seeds = [pa.SeedSequence().entropy for _ in range(3)]
    assert len(set(seeds)) == 3 and all(0 <= seed < 2**128 for seed in seeds)
    assert pa.default_rng().state != pa.default_rng().state
    assert pa.default_rng(None).state != pa.default_rng(None).state


DRAW = "import indrajala_math_rust as pa; print(pa.default_rng().random(4).tolist())"


def test_unseeded_generators_differ_between_processes():
    def draw() -> str:
        return subprocess.run([sys.executable, "-c", DRAW], capture_output=True, text=True, check=True).stdout

    assert draw() != draw()


def test_pickle_and_deepcopy_carry_the_state():
    rng = pa.default_rng(8)
    rng.random(9)
    for clone in [pickle.loads(pickle.dumps(rng)), copy.deepcopy(rng), copy.copy(rng)]:
        assert clone is not rng
        assert clone.state == rng.state
        assert_identical(clone.random(20), as_numpy(copy.deepcopy(rng).random(20)))
    reference = np.random.default_rng(8)
    reference.random(9)
    assert_identical(rng.random(20), reference.random(20))
