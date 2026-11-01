"""
The crate's erf (core/src/erf.rs, a port of fdlibm's s_erf.c through the correctly rounded exp;
indrajala-ml's FFN activations workplan, D2) is within one ulp of the exact erf, odd, and right at
its special values. The reference is Python's decimal at 100 significant digits: erf's Maclaurin
series, whose largest term at |x| = 6.5 is under 10^19, so 100 digits leave 80 after cancellation.
"""

import math
import random
import struct
from decimal import Decimal, localcontext

import pytest

from indrajala_math_rust import Array, erf

PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803")

# fdlibm's range boundaries (0.84375, 1.25, 1/0.35, 6) and 2^-28, each with its neighbours
BOUNDARIES = [0x3FEB000000000000, 0x3FF4000000000000, 0x4006DB6E00000000, 0x4018000000000000, 0x3E30000000000000]


def _exact_erf(x: float) -> float:
    with localcontext() as context:
        context.prec = 100
        d = Decimal(x)
        square = d * d
        term, total, n = d, d, 0
        while True:
            n += 1
            term = -term * square / n
            step = term / (2 * n + 1)
            total += step
            if step == 0 or abs(step) < abs(total) * Decimal("1e-90"):
                break
        return float(total * 2 / PI.sqrt())


def _erf(values: list[float]) -> list[float]:
    return erf(Array(values)).tolist()


def _from_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _inputs() -> list[float]:
    rng = random.Random(20261101)
    values = [rng.uniform(-6.5, 6.5) for _ in range(3000)]
    values += [math.copysign(10.0 ** rng.uniform(-30.0, 0.0), rng.choice((-1.0, 1.0))) for _ in range(500)]
    for bits in BOUNDARIES:
        for b in range(bits - 2, bits + 3):
            values += [_from_bits(b), -_from_bits(b)]
    return values


def test_within_one_ulp_of_the_exact_erf() -> None:
    values = _inputs()
    worst = 0.0
    for x, got in zip(values, _erf(values), strict=True):
        exact = _exact_erf(x)
        error = abs(got - exact) / math.ulp(exact)
        assert error <= 1.0, f"erf({x!r}) = {got!r}, exact {exact!r}: {error} ulp"
        worst = max(worst, error)
    assert worst <= 1.0


def test_odd() -> None:
    values = _inputs()
    assert _erf([-x for x in values]) == [-y for y in _erf(values)]


@pytest.mark.parametrize(
    ("x", "expected"),
    [(0.0, 0.0), (6.0, 1.0), (-6.0, -1.0), (27.0, 1.0), (math.inf, 1.0), (-math.inf, -1.0)],
)
def test_special_values(x: float, expected: float) -> None:
    assert _erf([x]) == [expected]


def test_signed_zero_and_nan() -> None:
    zero, negative_zero, nan = _erf([0.0, -0.0, math.nan])
    assert math.copysign(1.0, zero) == 1.0
    assert math.copysign(1.0, negative_zero) == -1.0
    assert math.isnan(nan)


def test_tiny_inputs() -> None:
    # |x| < 2^-28 is x + efx x, and below the smallest normal 0.125 (8 x + efx8 x), to avoid underflow
    values = [5e-324, 1e-310, 2.2e-308, 1e-20]
    for x, got in zip(values, _erf(values), strict=True):
        exact = _exact_erf(x)
        assert abs(got - exact) <= math.ulp(exact), f"erf({x!r}) = {got!r}, exact {exact!r}"
