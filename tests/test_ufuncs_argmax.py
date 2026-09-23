"""
argmax matches numpy's own tie-breaking rule (first occurrence on a tie), checked with a
randomized sweep plus an explicit tied-maximum case.
"""

import random

import numpy as np
import pytest

from indrajala_ml_array import Array, argmax


@pytest.mark.parametrize("seed", range(20))
def test_argmax_matches_numpy_across_a_random_sweep(seed):
    rng = random.Random(seed)
    data = [rng.uniform(-10.0, 10.0) for _ in range(9)]
    assert argmax(Array(data)) == int(np.argmax(np.array(data)))


def test_argmax_breaks_a_tie_by_first_occurrence_matching_numpy():
    data = [1.0, 3.0, 3.0, 2.0]
    assert argmax(Array(data)) == int(np.argmax(np.array(data))) == 1


def test_argmax_of_a_single_element_array():
    assert argmax(Array([5.0])) == 0


def test_argmax_rejects_empty_array():
    with pytest.raises(ValueError):
        argmax(Array.zeros(0))


def test_argmax_rejects_2d_array():
    with pytest.raises(ValueError):
        argmax(Array.zeros((2, 2)))
