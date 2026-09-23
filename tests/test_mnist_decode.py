"""
decode_mnist_pixels matches load_mnist_dataset_as_array's real output on a real MNIST sample,
exact match expected here - integer-to-float conversion and division have no RNG-style
irreproducibility, unlike uniform().
"""

import numpy as np
import pytest

from indrajala_ml_array import decode_mnist_pixels

from indrajala_ml.mnist_data import RECORD_SIZE, load_mnist_dataset_as_array

MNIST_TEST_PATH = "data/mnist/mnist-test.bin"


def _to_numpy(arr):
    rows, cols = arr.shape
    return np.array([[arr[r, c] for c in range(cols)] for r in range(rows)])


def test_decode_matches_load_mnist_dataset_as_array_on_a_real_sample():
    with open(MNIST_TEST_PATH, "rb") as f:
        raw = f.read(RECORD_SIZE * 50)

    actual = decode_mnist_pixels(raw, RECORD_SIZE)
    expected = load_mnist_dataset_as_array(MNIST_TEST_PATH, limit=50)

    assert actual.shape == expected.shape
    assert _to_numpy(actual) == pytest.approx(expected, abs=1e-15)


def test_decode_rejects_a_buffer_that_is_not_a_record_multiple():
    with pytest.raises(ValueError):
        decode_mnist_pixels(b"\x00" * (RECORD_SIZE + 1), RECORD_SIZE)


def test_decode_a_hand_constructed_single_record():
    # 3 pixel bytes + 1 label byte, record_size=4 - a small, fully-traceable case independent of
    # the real MNIST file, pinning the exact /255.0 normalization and label-byte exclusion.
    record = bytes([0, 128, 255, 7])  # last byte (7) is the label, dropped
    decoded = decode_mnist_pixels(record, record_size=4)
    assert decoded.shape == (1, 3)
    assert decoded[0, 0] == pytest.approx(0.0)
    assert decoded[0, 1] == pytest.approx(128 / 255.0)
    assert decoded[0, 2] == pytest.approx(1.0)
