"""
max_pool_forward_batch/max_pool_downstream_batch. Checked against a brute-force numpy reference
written from the definition of strided max pooling - a loop over every window, taking the first
maximal slot in row-major order and routing the window's delta to it - independent of
indrajala-ml's numpy MaxPoolArrayLayer. The forward pass and argmax involve no arithmetic, so they
must match exactly. The downstream scatter-add sums an input's deltas in slot order (numpy's
order), and this reference in window order, so with overlapping windows they can differ in the
last ULP.
"""

import itertools

import numpy as np
import pytest

from indrajala_math_rust import Array, ConvGeometry, max_pool_downstream_batch, max_pool_forward_batch

# (input_height, input_width, input_channels, pool_size, stride): non-overlapping, overlapping
# (stride < pool_size), gapped (stride > pool_size), multi-channel, non-square. The forward pass
# has a separate kernel for 2x2 at stride 2, so it gets even and odd sides (odd drops the last
# row and column); the rest go through the general kernel.
SHAPES = [
    (4, 4, 1, 2, 2),
    (6, 6, 2, 2, 2),
    (5, 7, 2, 2, 2),
    (5, 7, 3, 3, 3),
    (7, 6, 1, 3, 2),
    (5, 5, 2, 3, 1),
    (6, 5, 1, 2, 1),
    (7, 7, 2, 2, 3),
]
BATCH_SIZE = 3


def _np(arr):
    return np.array(arr.tolist())


def _geometry(shape):
    return ConvGeometry(*shape)


def _reference(X, delta, shape):
    """(A, argmax, dX) by brute force."""
    height, width, channels, p, s = shape
    out_h, out_w = (height - p) // s + 1, (width - p) // s + 1
    planes = X.reshape(len(X), channels, height, width)
    D = delta.reshape(len(X), channels, out_h, out_w)
    A = np.zeros((len(X), channels, out_h, out_w))
    argmax = np.zeros((len(X), channels, out_h, out_w))
    dX = np.zeros_like(planes)
    for n, c, out_row, out_col in itertools.product(range(len(X)), range(channels), range(out_h), range(out_w)):
        window = [planes[n, c, out_row * s + pr, out_col * s + pc] for pr in range(p) for pc in range(p)]
        slot = window.index(max(window))
        A[n, c, out_row, out_col] = window[slot]
        argmax[n, c, out_row, out_col] = slot
        dX[n, c, out_row * s + slot // p, out_col * s + slot % p] += D[n, c, out_row, out_col]
    return A.reshape(len(X), -1), argmax.reshape(len(X), -1), dX.reshape(len(X), -1)


def _inputs(rng, shape, tie_heavy):
    height, width, channels, _p, _s = shape
    size = (BATCH_SIZE, channels * height * width)
    if tie_heavy:
        # half exact zeros (what a ReLU layer produces), so ties inside a window are routine
        return rng.choice([0.0, 0.0, 0.0, 0.25, 0.5, 1.0], size=size)
    return rng.uniform(-1.0, 1.0, size=size)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tie_heavy", [False, True])
def test_max_pool_ops_match_the_brute_force_definition(shape, tie_heavy):
    rng = np.random.default_rng(0)
    g = _geometry(shape)
    X = _inputs(rng, shape, tie_heavy)
    delta = rng.uniform(-1.0, 1.0, size=(BATCH_SIZE, shape[2] * g.positions))
    expected_A, expected_argmax, expected_dX = _reference(X, delta, shape)

    A, argmax = max_pool_forward_batch(Array(X.tolist()), g)
    np.testing.assert_array_equal(_np(A), expected_A)
    np.testing.assert_array_equal(_np(argmax), expected_argmax)

    dX = max_pool_downstream_batch(Array(delta.tolist()), argmax, g)
    np.testing.assert_allclose(_np(dX), expected_dX, rtol=0, atol=1e-15)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tie_heavy", [False, True])
def test_a_vector_operand_is_one_example_with_the_same_bits(shape, tie_heavy):
    rng = np.random.default_rng(3)
    x, g = _inputs(rng, shape, tie_heavy)[0].tolist(), _geometry(shape)

    A_row, argmax_row = max_pool_forward_batch(Array([x]), g)
    d = rng.uniform(-1.0, 1.0, size=A_row.shape[1]).tolist()
    A_vec, argmax_vec = max_pool_forward_batch(Array(x), g)
    assert A_vec.shape == argmax_vec.shape == (A_row.shape[1],)
    assert A_vec.tolist() == A_row.tolist()[0]
    assert argmax_vec.tolist() == argmax_row.tolist()[0]

    dX_row = max_pool_downstream_batch(Array([d]), argmax_row, g)
    dX_vec = max_pool_downstream_batch(Array(d), argmax_vec, g)
    assert dX_vec.shape == (g.input_size,)
    assert dX_vec.tolist() == dX_row.tolist()[0]


@pytest.mark.parametrize("shape", SHAPES)
def test_signed_zero_ties_keep_the_first_slot_and_its_sign(shape):
    # -0.0 == 0.0, so a strict > keeps whichever zero comes first, sign included;
    # assert_array_equal can't tell the two zeros apart, so compare the bits
    rng = np.random.default_rng(5)
    height, width, channels, _p, _s = shape
    X = rng.choice([-0.0, 0.0, 0.0, -0.5], size=(BATCH_SIZE, channels * height * width))
    delta = np.zeros((BATCH_SIZE, channels * _geometry(shape).positions))
    expected_A, expected_argmax, _ = _reference(X, delta, shape)

    A, argmax = max_pool_forward_batch(Array(X.tolist()), _geometry(shape))
    assert np.signbit(expected_A).any() and not np.signbit(expected_A).all()
    np.testing.assert_array_equal(_np(A).view(np.int64), expected_A.view(np.int64))
    np.testing.assert_array_equal(_np(argmax), expected_argmax)


def test_ties_resolve_to_the_first_slot():
    g = ConvGeometry(2, 4, 1, 2, 2)  # two 2x2 windows side by side
    # left window all zero; right window a partial tie between slots 1 and 2 (row-major)
    X = [[0.0, 0.0, 0.1, 0.7, 0.0, 0.0, 0.7, 0.3]]
    A, argmax = max_pool_forward_batch(Array(X), g)
    assert A.tolist() == [[0.0, 0.7]]
    assert argmax.tolist() == [[0.0, 1.0]]

    dX = max_pool_downstream_batch(Array([[2.0, 3.0]]), argmax, g)
    assert dX.tolist() == [[2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0]]


def test_overlapping_windows_accumulate_on_a_shared_winner():
    # 3 wide, pool 2, stride 1: the centre column is in both windows and wins both
    g = ConvGeometry(2, 3, 1, 2, 1)
    X = [[0.0, 5.0, 0.0, 0.0, 0.0, 0.0]]
    A, argmax = max_pool_forward_batch(Array(X), g)
    assert A.tolist() == [[5.0, 5.0]]
    assert argmax.tolist() == [[1.0, 0.0]]
    dX = max_pool_downstream_batch(Array([[0.25, 0.5]]), argmax, g)
    assert dX.tolist() == [[0.0, 0.75, 0.0, 0.0, 0.0, 0.0]]


def test_max_pool_ops_reject_mismatched_shapes_and_bad_argmax():
    g = ConvGeometry(4, 4, 1, 2, 2)  # input_size 16, 4 windows
    with pytest.raises(ValueError):
        max_pool_forward_batch(Array.zeros((2, 15)), g)
    with pytest.raises(ValueError):
        max_pool_forward_batch(Array.zeros(15), g)  # a single-example vector of the wrong length
    with pytest.raises(ValueError):
        max_pool_downstream_batch(Array.zeros(5), Array.zeros(5), g)
    with pytest.raises(ValueError):
        max_pool_downstream_batch(Array.zeros(4), Array.zeros((1, 4)), g)  # mixed ranks
    with pytest.raises(ValueError):
        max_pool_downstream_batch(Array.zeros((1, 4)), Array.zeros(4), g)
    with pytest.raises(ValueError):
        max_pool_downstream_batch(Array.zeros((2, 5)), Array.zeros((2, 5)), g)
    with pytest.raises(ValueError):
        max_pool_downstream_batch(Array.zeros((2, 4)), Array.zeros((3, 4)), g)
    for bad in (4.0, -1.0, 0.5):
        with pytest.raises(ValueError):
            max_pool_downstream_batch(Array.zeros((1, 4)), Array([[0.0, 0.0, 0.0, bad]]), g)
