"""
uniform() cannot be checked for bit-identical parity against numpy's Mersenne Twister (a
hand-rolled PRNG never reproduces it) - so this checks range bounds, shape, and statistical
properties (mean/variance within tolerance across a large N) instead, a deliberately weaker bar
than this crate's other exact-match tests.
"""

import statistics

from indrajala_ml_array import uniform


def test_uniform_respects_shape_1d_and_2d():
    vector = uniform(-1.0, 1.0, 10)
    assert vector.shape == (10,)

    matrix = uniform(-1.0, 1.0, (3, 4))
    assert matrix.shape == (3, 4)


def test_uniform_draws_stay_within_the_requested_range():
    low, high = -2.0, 5.0
    arr = uniform(low, high, 5000)
    for i in range(5000):
        value = arr[i]
        assert low <= value < high


def test_uniform_mean_and_variance_are_statistically_plausible():
    # a uniform[low, high) distribution has mean (low+high)/2 and variance (high-low)^2/12 -
    # checked within a generous tolerance across a large N, not per-draw equality (see this
    # module's own docstring for why bit-identical parity isn't the achievable bar here).
    low, high = -3.0, 3.0
    n = 20000
    arr = uniform(low, high, n)
    values = [arr[i] for i in range(n)]

    expected_mean = (low + high) / 2.0
    expected_variance = (high - low) ** 2 / 12.0

    actual_mean = statistics.fmean(values)
    actual_variance = statistics.pvariance(values)

    assert abs(actual_mean - expected_mean) < 0.05
    assert abs(actual_variance - expected_variance) < 0.1


def test_consecutive_calls_do_not_repeat_the_same_draws():
    first = uniform(0.0, 1.0, 20)
    second = uniform(0.0, 1.0, 20)
    assert [first[i] for i in range(20)] != [second[i] for i in range(20)]


def test_fan_in_aware_randomize_formula_produces_a_usable_weight_matrix():
    # layer.W = np.random.uniform(-limit, limit, size=(size, previous_size)) - randomize()'s own
    # formula, reproduced here to confirm the shape and bound contract it depends on holds.
    previous_size = 16
    size = 8
    limit = 1.0 / previous_size**0.5

    w = uniform(-limit, limit, (size, previous_size))
    assert w.shape == (size, previous_size)
    for row in range(size):
        for col in range(previous_size):
            assert -limit <= w[row, col] < limit
