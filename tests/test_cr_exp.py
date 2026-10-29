"""
The crate's exp is correctly rounded (src/cr_exp.rs, a port of CORE-MATH's cr_exp; indrajala-ml's
aarch64 workplan, D4): every result is the binary64 value nearest the exact e^x, ties to even.
The reference is Python's decimal at 60 significant digits, rounded to binary64 by float() (Python
converts a decimal string correctly rounded): exp's hardest cases sit about 2^-120 from a rounding
boundary, far inside 60 digits, so the reference can't double-round.
"""

import math
import os
import random
import struct
from decimal import Decimal, localcontext

import numpy as np
import pytest

from indrajala_math_rust import Array, exp, exp_lanes, exp_scalar

# the inputs CORE-MATH's accurate path alone doesn't round (its database)
HARD_CASES = [
    "-0x1.8f80e06f3a04cp+4", "-0x1.daf3afee51991p+2", "-0x1.07e05e6e03b83p-2", "-0x1.daf693d64fadap-4",
    "-0x1.a8f783d749a8fp-4", "-0x1.40a990279c2cp-6", "-0x1.a2fefefd580dfp-13", "-0x1.664716b68a409p-14",
    "-0x1.0401ae48409b5p-28", "-0x1.ff171507f8ba5p-30", "-0x1.85e60704a3a9cp-30", "-0x1.4bd46601ae1efp-31",
    "0x1p-53", "0x1.ffff7fffe0001p-36", "0x1.8cd99fffb319cp-34", "0x1.413f33fe6ce07p-31",
    "0x1.f8e165f8388f7p-30", "0x1.282aae3169ee7p-20", "0x1.e5ba92aa9b1efp-20", "0x1.6a4d1af9cc989p-8",
    "0x1.aca7ae8da5a7bp+0",
]  # fmt: skip


def _bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _exact_exp(x: float) -> float:
    with localcontext() as context:
        context.prec = 60
        context.Emin, context.Emax = -999999, 999999
        return float(Decimal(x).exp())


def _check(values: list[float]) -> None:
    actual = exp(Array(values)).tolist()
    wrong = [(x, a, _exact_exp(x)) for x, a in zip(values, actual) if _bits(a) != _bits(_exact_exp(x))]
    assert wrong == [], [(x.hex(), a.hex(), e.hex()) for x, a, e in wrong[:5]]


RANGES = [
    (-20.0, 20.0, 20_000),  # where the networks' logits live
    (-745.2, 709.8, 20_000),  # the whole range, past both overflows
    (-745.14, -708.39, 5_000),  # subnormal results
    (-1e-3, 1e-3, 5_000),  # near 1
]


@pytest.mark.parametrize(("low", "high", "count"), RANGES)
def test_exp_is_correctly_rounded_on_a_random_sample(low: float, high: float, count: int):
    rng = random.Random(f"cr_exp {low} {high}")
    _check([rng.uniform(low, high) for _ in range(count)])


def test_exp_is_correctly_rounded_at_tiny_arguments_and_across_binades():
    rng = random.Random("cr_exp binades")
    values = [math.ldexp(rng.uniform(1.0, 2.0), e) * rng.choice((-1.0, 1.0)) for e in range(-60, 10) for _ in range(50)]
    _check(values + [2.0**-54, -(2.0**-54), 2.0**-53, 5e-324, -5e-324])


def test_exp_is_correctly_rounded_at_cr_exps_hard_cases():
    # the inputs CORE-MATH's accurate path alone doesn't round (its database), and their neighbours
    database = [float.fromhex(h) for h in HARD_CASES]
    _check(
        database + [math.nextafter(x, math.inf) for x in database] + [math.nextafter(x, -math.inf) for x in database]
    )


def test_exp_at_the_overflow_and_underflow_edges_and_special_values():
    largest = float.fromhex("0x1.62e42fefa39efp+9")  # the largest x with a finite exp
    edges = [largest, math.nextafter(largest, math.inf), -745.1332191019411, -745.1332191019412, -708.3964185322641]
    _check(edges + [math.nextafter(x, s) for x in edges for s in (math.inf, -math.inf)])
    special = exp(Array([math.inf, -math.inf, 0.0, -0.0, 1000.0, -1000.0])).tolist()
    assert special == [math.inf, 0.0, 1.0, 1.0, math.inf, 0.0]
    assert math.isnan(exp(Array([math.nan]))[0])


# The vector lanes (cr_exp::exp_into, indrajala-ml's crate kernels workplan, D1): `exp` above runs
# them; `exp_scalar` is one value at a time. The two must agree bit for bit everywhere, at every
# offset within a vector (4 lanes on AVX2, 2 on NEON) and in the tail.

LANE_SAMPLE = 2 ** int(os.environ.get("EXP_LANES_LOG2", "20"))  # 24 for the per-machine run


def _same_bits(values: list[float]) -> None:
    scalar = exp_scalar(Array(values)).tolist()
    for vector in (exp(Array(values)).tolist(), exp_lanes(Array(values))[0].tolist()):  # into, in place
        wrong = [(x, v, s) for x, v, s in zip(values, vector, scalar) if _bits(v) != _bits(s)]
        assert wrong == [], [(x.hex(), v.hex(), s.hex()) for x, v, s in wrong[:5]]


def _every_offset(values: list[float]) -> list[list[float]]:
    # each value at each of a vector's lanes, and in a tail: shifted by 0-4 places
    return [[0.5] * shift + values for shift in range(5)]


def test_the_vector_lanes_match_the_scalar_exp_on_every_test_input():
    rng = random.Random("cr_exp lanes")
    database = [float.fromhex(h) for h in HARD_CASES]
    largest = float.fromhex("0x1.62e42fefa39efp+9")
    values = (
        [rng.uniform(low, high) for low, high, _ in RANGES for _ in range(2_000)]
        + [math.ldexp(rng.uniform(1.0, 2.0), e) * rng.choice((-1.0, 1.0)) for e in range(-60, 10) for _ in range(5)]
        + [2.0**-54, -(2.0**-54), 2.0**-53, 5e-324, -5e-324]
        + database
        + [math.nextafter(x, s) for x in database for s in (math.inf, -math.inf)]
        + [largest, math.nextafter(largest, math.inf), -745.1332191019411, -745.1332191019412, -708.3964185322641]
        + [math.inf, -math.inf, 0.0, -0.0, 1000.0, -1000.0, math.nan]
    )
    for shifted in _every_offset(values):
        _same_bits(shifted)


def test_the_vector_lanes_match_the_scalar_exp_at_the_ordinary_ranges_edges():
    # the lanes' range test: |x| just past 2^-54 and just short of the subnormal results
    edges = [2.0**-54, float.fromhex("-0x1.6232bdd7abcd2p+9"), float.fromhex("0x1.6232bdd7abcd2p+9")]
    near = [x for e in edges for x in (e, math.nextafter(e, math.inf), math.nextafter(e, -math.inf), -e)]
    for shifted in _every_offset(near):
        _same_bits(shifted)


def test_the_vector_lanes_match_the_scalar_exp_on_random_bits_and_the_range():
    rng = np.random.default_rng(20261029)
    every_double = rng.integers(0, 2**64, size=LANE_SAMPLE // 2, dtype=np.uint64).view(np.float64)
    in_range = rng.uniform(-746.0, 710.0, size=LANE_SAMPLE // 2)
    values = np.concatenate([every_double, in_range])
    scalar = np.array(exp_scalar(Array(values.tolist())).tolist()).view(np.uint64)
    nan = np.isnan(values)  # a NaN's payload isn't claimed (the README): only that it is NaN
    for result in (exp(Array(values.tolist())), exp_lanes(Array(values.tolist()))[0]):  # into, in place
        vector = np.array(result.tolist()).view(np.uint64)
        assert np.isnan(vector.view(np.float64)[nan]).all()
        wrong = np.flatnonzero((vector != scalar) & ~nan)
        assert wrong.size == 0, [float(values[i]).hex() for i in wrong[:5]]


def test_the_vector_lanes_redo_only_the_rare_values():
    rng = random.Random("cr_exp redone")
    ordinary = [rng.uniform(-20.0, 20.0) for _ in range(400_000)]
    redone = exp_lanes(Array(ordinary))[1]
    # the fast path rounds about 689 in 690 values (CORE-MATH's figure for cr_exp)
    assert 0 < redone < len(ordinary) / 200
    # the hard cases' accurate path is reached through the lanes, and its results are right
    database = [float.fromhex(h) for h in HARD_CASES]
    assert exp_lanes(Array(database * 4))[1] > 0
    _check(database * 4)
