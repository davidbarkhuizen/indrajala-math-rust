"""
The Python<->Rust round-trip works for construction (including from_rows), shape, single-element read/write (both the
1D scalar-index and 2D tuple-index shapes), .copy(), .reshape(), and the row ops row()/take_rows().
"""

import pytest

from indrajala_math_rust import Array


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


def test_from_rows_matches_nested_construction_for_lists_and_tuples():
    nested = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    expected = Array(nested).tolist()
    assert Array.from_rows(nested).tolist() == expected
    assert Array.from_rows([tuple(row) for row in nested]).tolist() == expected
    assert Array.from_rows(tuple(tuple(row) for row in nested)).shape == (2, 3)


def test_from_rows_rejects_ragged_and_empty_input():
    with pytest.raises(ValueError):
        Array.from_rows([[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError):
        Array.from_rows([[1.0], [2.0, 3.0]])
    with pytest.raises(ValueError):
        Array.from_rows([])
    with pytest.raises(TypeError):
        Array.from_rows([[1.0, "x"]])


def test_row_copies_one_row_as_a_vector():
    arr = Array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    row = arr.row(1)
    assert row.shape == (2,)
    assert row.tolist() == [3.0, 4.0]
    row[0] = 99.0
    assert arr[1, 0] == 3.0
    with pytest.raises(IndexError):
        arr.row(3)
    with pytest.raises(OverflowError):
        arr.row(-1)
    with pytest.raises(ValueError):
        Array([1.0, 2.0]).row(0)


def test_take_rows_gathers_rows_in_the_given_order_with_repeats():
    arr = Array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    taken = arr.take_rows([2, 0, 2])
    assert taken.shape == (3, 2)
    assert taken.tolist() == [[5.0, 6.0], [1.0, 2.0], [5.0, 6.0]]
    with pytest.raises(IndexError):
        arr.take_rows([0, 3])
    with pytest.raises(ValueError):
        arr.take_rows([])
    with pytest.raises(ValueError):
        Array([1.0, 2.0]).take_rows([0])


def test_from_rows_reads_other_sequences_through_iteration():
    from array import array

    rows = [array("d", [1.0, 2.0]), array("d", [3.0, 4.0])]
    assert Array.from_rows(rows).tolist() == [[1.0, 2.0], [3.0, 4.0]]
