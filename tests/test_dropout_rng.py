"""
bernoulli_mask/draw_bernoulli_mask is an RNG primitive, a fresh category from the mechanical
fused-arithmetic ops elsewhere in this crate. Checked directly against numpy's own
np.random.random(shape) >= drop_probability formula before DropoutRustArrayLayer/
layer_dropout_forward ever rely on it, the same "prove the primitive against numpy before
building the layer" discipline array_relu/array_softmax follow. Like uniform() (test_random_uniform.py), bit-identical parity against numpy's own
Mersenne Twister stream isn't achievable with a hand-rolled generator - this checks range/shape and
statistical keep-rate instead, not per-draw equality.
"""

import statistics

from indrajala_ml_array import bernoulli_mask


def test_bernoulli_mask_respects_shape_1d_and_2d():
    vector = bernoulli_mask(0.5, 10)
    assert vector.shape == (10,)

    matrix = bernoulli_mask(0.5, (3, 4))
    assert matrix.shape == (3, 4)


def test_bernoulli_mask_entries_are_exactly_zero_or_one():
    arr = bernoulli_mask(0.5, 2000)
    for i in range(2000):
        assert arr[i] in (0.0, 1.0)


def test_bernoulli_mask_mean_keep_rate_is_statistically_plausible():
    # a Bernoulli(keep_probability) mask has mean keep_probability - checked within a generous
    # tolerance across a large N, not per-draw equality (see this module's own docstring).
    drop_probability = 0.3
    expected_keep_rate = 1.0 - drop_probability
    n = 20000

    arr = bernoulli_mask(drop_probability, n)
    values = [arr[i] for i in range(n)]

    actual_keep_rate = statistics.fmean(values)
    assert abs(actual_keep_rate - expected_keep_rate) < 0.02


def test_bernoulli_mask_drop_probability_zero_never_drops():
    arr = bernoulli_mask(0.0, 500)
    assert all(arr[i] == 1.0 for i in range(500))


def test_bernoulli_mask_drop_probability_near_one_almost_always_drops():
    arr = bernoulli_mask(0.999999, 500)
    assert sum(arr[i] for i in range(500)) < 5  # overwhelmingly dropped, not exactly zero kept


def test_consecutive_calls_do_not_repeat_the_same_draws():
    first = bernoulli_mask(0.5, 50)
    second = bernoulli_mask(0.5, 50)
    assert [first[i] for i in range(50)] != [second[i] for i in range(50)]
