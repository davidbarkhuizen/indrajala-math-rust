"""
.tolist() and construct-from-list (Array(...)) round-trip a 1D and a 2D array through
list-and-back with bit-identical recovery - the pair save()/load() rely on for JSON
serialization.
"""

import numpy as np

from indrajala_ml_array import Array


def test_1d_tolist_round_trip():
    data = [1.0, 2.5, -3.0, 0.0]
    arr = Array(data)
    assert arr.tolist() == data

    rebuilt = Array(arr.tolist())
    assert rebuilt.shape == arr.shape
    assert rebuilt.tolist() == data


def test_2d_tolist_returns_a_nested_list_of_lists():
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    arr = Array(data)
    assert arr.tolist() == data

    rebuilt = Array(arr.tolist())
    assert rebuilt.shape == (2, 3)
    assert rebuilt.tolist() == data


def test_tolist_matches_numpys_own_tolist():
    data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert Array(data).tolist() == np.array(data).tolist()

    vector_data = [1.0, 2.0, 3.0]
    assert Array(vector_data).tolist() == np.array(vector_data).tolist()


def test_snapshot_style_round_trip_survives_a_json_cycle(tmp_path):
    import json

    w = Array([[1.0, 2.0], [3.0, 4.0]])
    b = Array([0.5, -0.5])

    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"W": w.tolist(), "b": b.tolist()}))

    loaded = json.loads(path.read_text())
    restored_w = Array(loaded["W"])
    restored_b = Array(loaded["b"])

    assert restored_w.tolist() == w.tolist()
    assert restored_b.tolist() == b.tolist()
