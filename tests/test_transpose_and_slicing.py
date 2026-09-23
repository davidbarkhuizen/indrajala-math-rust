"""
.T (a no-op on 1D, a real transpose on 2D) and arr[:, :-1]-style contiguous slicing, checked
against a hand-constructed array with a known, distinctive pattern, the same "is the indexing
convention right" discipline the convolutional layer's own hot-pixel test uses.
"""

import pytest

from indrajala_ml_array import Array


def test_transpose_of_1d_is_a_no_op():
    vector = Array([1.0, 2.0, 3.0])
    transposed = vector.T
    assert transposed.shape == (3,)
    assert [transposed[i] for i in range(3)] == [1.0, 2.0, 3.0]


def test_transpose_of_2d_swaps_axes():
    matrix = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    transposed = matrix.T
    assert transposed.shape == (3, 2)
    expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    for row in range(3):
        for col in range(2):
            assert transposed[row, col] == expected[row][col]


def test_transpose_of_transpose_round_trips():
    matrix = Array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert [
        matrix.T.T[row, col] for row in range(3) for col in range(2)
    ] == [matrix[row, col] for row in range(3) for col in range(2)]


def _hot_pixel_matrix(rows: int, cols: int, hot_row: int, hot_col: int) -> Array:
    grid = [[0.0] * cols for _ in range(rows)]
    grid[hot_row][hot_col] = 1.0
    return Array(grid)


def test_full_row_slice_with_trailing_column_dropped_matches_mnist_shape():
    # mirrors load_mnist_dataset_as_array's records[:, :-1]: every row, all but the last column
    matrix = _hot_pixel_matrix(rows=4, cols=5, hot_row=2, hot_col=4)
    sliced = matrix[:, :-1]
    assert sliced.shape == (4, 4)
    # the hot pixel was in the dropped last column, so the slice is now all zeros
    assert all(
        sliced[row, col] == 0.0 for row in range(4) for col in range(4)
    )


def test_slice_preserves_a_hot_pixel_still_inside_the_kept_range():
    matrix = _hot_pixel_matrix(rows=4, cols=5, hot_row=1, hot_col=2)
    sliced = matrix[:, :-1]
    assert sliced.shape == (4, 4)
    for row in range(4):
        for col in range(4):
            expected = 1.0 if (row, col) == (1, 2) else 0.0
            assert sliced[row, col] == expected


def test_slice_with_explicit_start_and_stop():
    matrix = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    sliced = matrix[1:3, 0:2]
    assert sliced.shape == (2, 2)
    assert [sliced[r, c] for r in range(2) for c in range(2)] == [4.0, 5.0, 7.0, 8.0]


def test_stepped_slice_is_rejected():
    matrix = Array.zeros((4, 4))
    with pytest.raises(ValueError):
        matrix[::2, :]
