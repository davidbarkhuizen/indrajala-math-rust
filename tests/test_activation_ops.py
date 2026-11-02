"""
The hidden layers' activations (core/src/activation.rs; indrajala-ml's FFN activations workplan,
D9): ReLU, GELU (exact, through erf) and GELU-tanh (GPT-2's), in the fused dense ops
(`layer_activation_*`) and the mask (`array_activation_mask`).

Three kinds of check:

- **By bits, the grouping:** each op against numpy's elementwise `+ - * /` (correctly rounded) in
  the module doc's grouping, with the crate's own `erf` and `exp` ufuncs for the transcendentals
  and the crate's own unfused products (`affine_forward*`, `layer_downstream*`), so the bits must
  agree. This is what indrajala-ml's numpy and pure-Python layers must match.
- **Against the exact functions:** a `decimal` reference at 60 digits (erf's Maclaurin series;
  tanh through `Decimal.exp`), within a few ulps plus the cancellation `1 + erf` and `1 + tanh`
  allow for negative `z`.
- **The ReLU arm** keeps no `z` and masks on `a > 0`, as the ReLU ops it replaces did.
"""

import math
import random
from decimal import Decimal, localcontext
from typing import Literal

import numpy as np
import pytest
from array_helpers import bits, to_array, to_numpy

from indrajala_math_rust import (
    Array,
    affine_forward,
    affine_forward_batch,
    array_activation_mask,
    erf,
    exp,
    layer_activation_forward,
    layer_activation_forward_batch,
    layer_activation_hidden_delta,
    layer_activation_hidden_delta_batch,
    layer_activation_hidden_delta_skip,
    layer_activation_hidden_delta_skip_batch,
    layer_downstream,
    layer_downstream_batch,
)

SQRT1_2 = 0.7071067811865476
INV_SQRT_2PI = 0.3989422804014327
C = 0.7978845608028654
K = 0.044715
type Name = Literal["relu", "gelu", "gelu_tanh"]

GELUS: list[Name] = ["gelu", "gelu_tanh"]
ACTIVATIONS: list[Name] = ["relu", *GELUS]
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803")

# (batch, input_size, size): one example up to the threading threshold's (8M flops)
SHAPES = [(1, 1, 1), (3, 7, 5), (5, 13, 9), (32, 784, 30), (7, 5408, 32)]


def _erf(z: np.ndarray) -> np.ndarray:
    return to_numpy(erf(to_array(z))).reshape(z.shape)


def _exp(z: np.ndarray) -> np.ndarray:
    return to_numpy(exp(to_array(z))).reshape(z.shape)


def _th(z: np.ndarray) -> np.ndarray:
    # the cell's tanh, 1 - 2 / (exp(2u) + 1), of GPT-2's argument
    u = C * (z + K * ((z * z) * z))
    with np.errstate(over="ignore"):
        return 1.0 - 2.0 / (_exp(2.0 * u) + 1.0)


def _value(activation: Name, z: np.ndarray) -> np.ndarray:
    if activation == "relu":
        return np.where(z > 0.0, z, 0.0)
    if activation == "gelu":
        return (0.5 * z) * (1.0 + _erf(z * SQRT1_2))
    return (0.5 * z) * (1.0 + _th(z))


def _derivative(activation: Name, z: np.ndarray) -> np.ndarray:
    if activation == "gelu":
        return 0.5 * (1.0 + _erf(z * SQRT1_2)) + z * (_exp(-0.5 * (z * z)) * INV_SQRT_2PI)
    th = _th(z)
    return 0.5 * (1.0 + th) + (0.5 * z) * ((1.0 - th * th) * (C * (1.0 + (3.0 * K) * (z * z))))


def _mask(activation: Name, downstream: np.ndarray, cached: np.ndarray) -> np.ndarray:
    if activation == "relu":
        return np.where(cached > 0.0, downstream, 0.0)
    return downstream * _derivative(activation, cached)


def _scalar(activation: Name, values: list[float]) -> list[float]:
    # act(z) through the fused op itself: 1 * z + 0 is z exactly (but for -0.0, which becomes 0.0)
    rows = Array([[v] for v in values])
    a = layer_activation_forward_batch(Array([[1.0]]), rows, Array([0.0]), activation)[0]
    return [float(v) for v in np.ravel(to_numpy(a))]


def _scalar_derivative(activation: Name, values: list[float]) -> list[float]:
    return array_activation_mask(Array([1.0] * len(values)), Array(values), activation).tolist()


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("batch, input_size, size", SHAPES)
def test_the_forward_is_the_activation_of_the_affine_product(activation: Name, batch: int, input_size: int, size: int):
    rng = np.random.default_rng(batch * 1000 + size)
    w = to_array(rng.uniform(-1.0, 1.0, (size, input_size)))
    x = rng.uniform(-1.0, 1.0, (batch, input_size)) * 4.0 / math.sqrt(input_size)
    b = to_array(rng.uniform(-1.0, 1.0, size))
    z = to_numpy(affine_forward_batch(w, to_array(x), b))

    a, kept = layer_activation_forward_batch(w, to_array(x), b, activation)

    assert bits(a) == bits(_value(activation, z))
    if activation == "relu":
        assert kept is None
    else:
        assert kept is not None and bits(kept) == bits(z)
    # the single-example op, row by row
    for row in range(batch):
        a_row, kept_row = layer_activation_forward(w, to_array(x[row]), b, activation)
        assert bits(a_row) == bits(to_numpy(a)[row])
        assert bits(to_numpy(affine_forward(w, to_array(x[row]), b))) == bits(z[row])
        assert (kept_row is None) == (activation == "relu")


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("rows", [None, 1, 4, 33])
@pytest.mark.parametrize("next_size, size", [(1, 1), (3, 2), (10, 64), (40, 256)])
def test_the_hidden_deltas_mask_the_downstream(activation: Name, rows: int | None, next_size: int, size: int):
    rng = np.random.default_rng(next_size * 100 + size)
    shape = (size,) if rows is None else (rows, size)
    next_shape = (next_size,) if rows is None else (rows, next_size)
    next_w = to_array(rng.uniform(-1.0, 1.0, (next_size, size)))
    next_delta = to_array(rng.uniform(-1.0, 1.0, next_shape))
    skip = rng.uniform(-1.0, 1.0, shape)
    cached = rng.uniform(-4.0, 4.0, shape)
    if activation == "relu":
        cached = np.abs(cached)
        cached[..., 0] = 0.0  # ReLU's boundary: inactive
    if rows is None:
        downstream = to_numpy(layer_downstream(next_w, next_delta))
        plain = layer_activation_hidden_delta(next_w, next_delta, to_array(cached), activation)
        skipped = layer_activation_hidden_delta_skip(next_w, next_delta, to_array(skip), to_array(cached), activation)
    else:
        downstream = to_numpy(layer_downstream_batch(next_w, next_delta))
        plain = layer_activation_hidden_delta_batch(next_w, next_delta, to_array(cached), activation)
        skipped = layer_activation_hidden_delta_skip_batch(
            next_w, next_delta, to_array(skip), to_array(cached), activation
        )

    assert bits(plain) == bits(_mask(activation, downstream, cached))
    assert bits(skipped) == bits(_mask(activation, downstream + skip, cached))
    assert bits(array_activation_mask(to_array(downstream), to_array(cached), activation)) == bits(plain)


def _decimal_erf(x: Decimal) -> Decimal:
    square = x * x
    term, total, n = x, x, 0
    while True:
        n += 1
        term = -term * square / n
        step = term / (2 * n + 1)
        total += step
        if step == 0 or abs(step) < abs(total) * Decimal("1e-55"):
            return total * 2 / PI.sqrt()


def _exact(activation: Name, z: float) -> tuple[float, float]:
    """(act(z), act'(z)) in 60 digits."""
    with localcontext() as context:
        context.prec = 60
        d = Decimal(z)
        if activation == "gelu":
            phi = (1 + _decimal_erf(d / Decimal(2).sqrt())) / 2
            density = (-(d * d) / 2).exp() / (2 * PI).sqrt()
            return float(d * phi), float(phi + d * density)
        c = (2 / PI).sqrt()
        k = Decimal("0.044715")
        u = c * (d + k * d * d * d)
        e = (2 * u).exp()
        th = (e - 1) / (e + 1)
        value = d / 2 * (1 + th)
        derivative = (1 + th) / 2 + d / 2 * (1 - th * th) * c * (1 + 3 * k * d * d)
        return float(value), float(derivative)


def _inputs() -> list[float]:
    rng = random.Random(20261102)
    values = [rng.uniform(-6.0, 6.0) for _ in range(1500)]
    values += [math.copysign(10.0 ** rng.uniform(-12.0, 0.0), rng.choice((-1.0, 1.0))) for _ in range(300)]
    return values


@pytest.mark.parametrize("activation", GELUS)
def test_the_gelus_are_within_a_few_ulps_of_the_exact_functions(activation: Name):
    values = _inputs()
    got_values = _scalar(activation, values)
    got_derivatives = _scalar_derivative(activation, values)
    for z, value, derivative in zip(values, got_values, got_derivatives, strict=True):
        exact_value, exact_derivative = _exact(activation, z)
        # a few ulps, plus 1 + erf's (1 + tanh's) cancellation for negative z: an ulp of 1, scaled
        assert abs(value - exact_value) <= 4 * math.ulp(exact_value) + abs(z) * 2.0**-52, (z, value, exact_value)
        tolerance = 4 * math.ulp(exact_derivative) + (1.0 + z * z) * 2.0**-52
        assert abs(derivative - exact_derivative) <= tolerance, (z, derivative, exact_derivative)


@pytest.mark.parametrize("activation", GELUS)
def test_the_gelus_at_their_limits(activation: Name):
    # large z: the identity and derivative 1; large negative z: 0 and derivative 0; z = 0: 0, 1/2
    values = [0.0, 40.0, -40.0, 1e6, -1e6]
    assert _scalar(activation, values) == [0.0, 40.0, -0.0, 1e6, -0.0]
    assert _scalar_derivative(activation, values) == [0.5, 1.0, 0.0, 1.0, 0.0]


def test_relu_masks_on_the_output_and_gives_positive_zero():
    downstream = Array([3.0, -3.0, 2.0, 5.0])
    a = Array([0.0, 0.0, 1.5, -0.0])
    assert bits(array_activation_mask(downstream, a, "relu")) == bits(np.array([0.0, 0.0, 2.0, 0.0]))
    assert _scalar("relu", [-2.0, 0.0, 2.5]) == [0.0, 0.0, 2.5]


def test_an_unknown_activation_is_refused():
    w, x, b = Array([[1.0]]), Array([1.0]), Array([0.0])
    with pytest.raises(ValueError, match="unknown activation"):
        layer_activation_forward(w, x, b, "swish")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown activation"):
        array_activation_mask(x, x, "Relu")  # type: ignore[arg-type]


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="matching shapes"):
        array_activation_mask(Array.zeros((2, 3)), Array.zeros((3, 2)), "gelu")
