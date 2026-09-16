use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};

/// Elementwise `e^x` over a whole array - mirrors `array_layer.sigmoid`'s own reliance on
/// `exp`'s overflow behavior (a large negative `z` drives `exp(-z)` to `f64::INFINITY`, and
/// `1.0 / (1.0 + f64::INFINITY) == 0.0` under IEEE 754) instead of `math.exp`'s
/// `OverflowError`-raising behavior in the pure-Python reference. See
/// docs/vectorized-array-classes.md's own "numerical parity validation" - this is the one
/// operation with a documented overflow-boundary trap already flagged twice over, checked here
/// directly against `exp` before `sigmoid` itself is ever built on top of it.
#[pyfunction]
pub fn exp(arr: &RustArray) -> RustArray {
    RustArray {
        data: arr.data.iter().map(|value| value.exp()).collect(),
        shape: arr.shape,
    }
}

/// Sums a 2D array's rows into a 1D vector - the one fixed-axis (`axis=0`) reduction
/// docs/numpy-interface-subset.md's table actually requires
/// (`accumulate_gradient_batch`'s batched bias gradient, `self._grad_b += self.delta_batch.sum(axis=0)`),
/// added after cross-checking the real implementation - see that document's own "re-checked
/// against the built implementation" note. General axis-parameterized reduction stays out of
/// scope; this is the one fixed case, not a general `axis=` parameter.
#[pyfunction]
pub fn sum_axis0(arr: &RustArray) -> PyResult<RustArray> {
    match arr.shape {
        Shape::Matrix(rows, cols) => {
            let mut out = vec![0.0; cols];
            for row in 0..rows {
                for col in 0..cols {
                    out[col] += arr.data[row * cols + col];
                }
            }
            Ok(RustArray::from_vector(out))
        }
        Shape::Vector(_) => Err(PyValueError::new_err(
            "sum_axis0 requires a 2D array, got a 1D vector",
        )),
    }
}

/// The index of the largest element in a 1D array - `classify_state`'s own
/// `np.argmax(self.predict_probabilities(state))`. Strict `>` (not `>=`) when scanning left to
/// right keeps the first occurrence on a tie, matching numpy's own `np.argmax` tie-breaking rule
/// - a real behavioral detail to match, not assume (see docs/rust-array-core.md's own "PR 6").
#[pyfunction]
pub fn argmax(arr: &RustArray) -> PyResult<usize> {
    match arr.shape {
        Shape::Vector(n) => {
            if n == 0 {
                return Err(PyValueError::new_err("argmax of an empty array"));
            }
            let mut best_index = 0;
            let mut best_value = arr.data[0];
            for i in 1..n {
                if arr.data[i] > best_value {
                    best_value = arr.data[i];
                    best_index = i;
                }
            }
            Ok(best_index)
        }
        Shape::Matrix(_, _) => Err(PyValueError::new_err("argmax requires a 1D array")),
    }
}
