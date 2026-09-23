"""
The PyO3/maturin toolchain works end to end - the extension builds, installs into the active
venv, and is importable and callable from Python.
No array type exists yet, so there's nothing to parity-check against numpy or pure Python here.
"""

import indrajala_ml_array


def test_ping_round_trips_through_rust():
    assert indrajala_ml_array.ping() == "pong"
