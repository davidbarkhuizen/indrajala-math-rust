"""
The Python<->Rust round-trip works for construction, shape, single-element read/write (both the
1D scalar-index and 2D tuple-index shapes), .copy(), and .reshape().
"""

import pytest

from indrajala_ml_array import Array


def test_construct_1d_from_flat_list_and_read_back():
    arr = Array([1.0, 2.0, 3.0])
    assert arr.shape == (3,)
    assert [arr[i] for i in range(3)] == [1.0, 2.0, 3.0]


def test_construct_2d_from_nested_list_and_read_back():
    arr = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert arr.shape == (2, 3)
    for row in range(2):
        for col in range(3):
            assert arr[row, col] == pytest.approx(row * 3 + col + 1)


def test_construct_rejects_ragged_rows():
    with pytest.raises(ValueError):
        Array([[1.0, 2.0], [3.0]])


def test_zeros_1d_and_2d():
    vector = Array.zeros(4)
    assert vector.shape == (4,)
    assert [vector[i] for i in range(4)] == [0.0, 0.0, 0.0, 0.0]

    matrix = Array.zeros((2, 3))
    assert matrix.shape == (2, 3)
    for row in range(2):
        for col in range(3):
            assert matrix[row, col] == 0.0


def test_setitem_1d_scalar_index():
    arr = Array.zeros(5)
    arr[2] = 1.0
    assert arr[2] == 1.0
    assert arr[0] == 0.0


def test_setitem_2d_tuple_index():
    arr = Array.zeros((3, 4))
    arr[1, 2] = 9.0
    assert arr[1, 2] == 9.0
    assert arr[0, 0] == 0.0
    assert arr[2, 3] == 0.0


def test_out_of_range_index_raises():
    vector = Array.zeros(3)
    with pytest.raises(IndexError):
        vector[3]

    matrix = Array.zeros((2, 2))
    with pytest.raises(IndexError):
        matrix[2, 0]
    with pytest.raises(IndexError):
        matrix[0, 2]


def test_copy_is_independent_of_the_original():
    original = Array([1.0, 2.0, 3.0])
    duplicate = original.copy()
    duplicate[0] = 99.0
    assert original[0] == 1.0
    assert duplicate[0] == 99.0


def test_reshape_preserves_data_and_order():
    flat = Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    matrix = flat.reshape((2, 3))
    assert matrix.shape == (2, 3)
    assert [matrix[row, col] for row in range(2) for col in range(3)] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]

    back_to_vector = matrix.reshape(6)
    assert back_to_vector.shape == (6,)
    assert [back_to_vector[i] for i in range(6)] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_reshape_rejects_a_size_mismatch():
    arr = Array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        arr.reshape((3, 3))
