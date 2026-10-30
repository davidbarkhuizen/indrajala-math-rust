"""
What the op tests share: an Array as numpy and back, and values by bits (0.0 and -0.0 differ, where
np.array_equal takes them as equal). Each test file imports them (pytest puts tests/ on sys.path).
"""

import numpy as np

from indrajala_math_rust import Array


def to_numpy(array: Array) -> np.ndarray:
    return np.array(array.tolist())


def to_array(values: np.ndarray) -> Array:
    return Array(values.tolist())


def bits(values: np.ndarray | Array) -> list[bytes]:
    """Every value's float64 bytes, in row-major order."""
    floats = np.array(values.tolist(), dtype=np.float64) if isinstance(values, Array) else values
    return [v.tobytes() for v in np.ravel(floats).astype(np.float64)]
