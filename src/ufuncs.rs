use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};
use crate::ops::same_shape_elementwise;

/// Elementwise `e^x` over a whole array - mirrors `array_layer.sigmoid`'s own reliance on
/// `exp`'s overflow behavior (a large negative `z` drives `exp(-z)` to `f64::INFINITY`, and
/// `1.0 / (1.0 + f64::INFINITY) == 0.0` under IEEE 754) instead of `math.exp`'s
/// `OverflowError`-raising behavior in the pure-Python reference - the one operation with a
/// documented overflow-boundary trap, checked here directly against `exp` before `sigmoid`
/// itself is ever built on top of it.
#[pyfunction]
pub fn exp(arr: &RustArray) -> RustArray {
    RustArray {
        data: arr.data.iter().map(|value| value.exp()).collect(),
        shape: arr.shape,
    }
}

/// Sums a 2D array's rows into a 1D vector - the one fixed-axis (`axis=0`) reduction this core
/// needs (`accumulate_gradient_batch`'s batched bias gradient,
/// `self._grad_b += self.delta_batch.sum(axis=0)`), cross-checked directly against that real
/// implementation. General axis-parameterized reduction stays out of scope; this is the one
/// fixed case, not a general `axis=` parameter.
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

/// Elementwise `max(0.0, x)` - the Rust core's counterpart to `np.maximum(0.0, z)`
/// (`ArrayLayer.forward`'s array-level `relu_activation`). `np.maximum`/`np.where` remain
/// otherwise out of scope for this core; this is the one exception.
#[pyfunction]
pub fn array_relu(arr: &RustArray) -> RustArray {
    RustArray {
        data: arr.data.iter().map(|&v| v.max(0.0)).collect(),
        shape: arr.shape,
    }
}

/// `downstream * (a > 0.0)` - ReLU's derivative mask, zeroing out the downstream gradient
/// wherever this layer's own cached activation was <= 0 (`ReLUArrayLayer.compute_hidden_delta`/
/// `compute_hidden_delta_batch`'s shared formula). Shape-agnostic (single-example 1D or batched
/// 2D), so one function covers both call shapes.
#[pyfunction]
pub fn array_relu_mask(downstream: &RustArray, a: &RustArray) -> PyResult<RustArray> {
    if downstream.shape != a.shape {
        return Err(PyValueError::new_err(format!(
            "array_relu_mask requires matching shapes, got {:?} and {:?}",
            downstream.shape, a.shape
        )));
    }
    Ok(RustArray {
        data: same_shape_elementwise(&downstream.data, &a.data, |d, av| if av > 0.0 { d } else { 0.0 }),
        shape: downstream.shape,
    })
}

/// Numerically-stable softmax normalization - `SoftmaxArrayLayer.forward`/`forward_batch`'s own
/// max-shift-then-exponentiate-then-normalize formula, validated directly against numpy
/// independently of either array-based softmax class, the same way `array_relu`/`array_relu_mask`
/// are validated for ReLU. A 1D vector is normalized as one whole distribution (`forward`'s
/// shape); a 2D matrix is normalized row-wise, one independent distribution per row
/// (`forward_batch`'s shape) - general axis-parameterized reduction stays out of scope; this is
/// the one fixed case, matching `sum_axis0`'s own precedent.
#[pyfunction]
pub fn array_softmax(arr: &RustArray) -> RustArray {
    fn normalize_row(row: &[f64]) -> Vec<f64> {
        let max_value = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = row.iter().map(|&v| (v - max_value).exp()).collect();
        let total: f64 = exp_values.iter().sum();
        exp_values.iter().map(|&e| e / total).collect()
    }

    match arr.shape {
        Shape::Vector(_) => RustArray {
            data: normalize_row(&arr.data),
            shape: arr.shape,
        },
        Shape::Matrix(rows, cols) => {
            let mut out = vec![0.0; rows * cols];
            for row in 0..rows {
                let start = row * cols;
                let normalized = normalize_row(&arr.data[start..start + cols]);
                out[start..start + cols].copy_from_slice(&normalized);
            }
            RustArray {
                data: out,
                shape: arr.shape,
            }
        }
    }
}

/// The index of the largest element in a 1D array - `classify_state`'s own
/// `np.argmax(self.predict_probabilities(state))`. Strict `>` (not `>=`) when scanning left to
/// right keeps the first occurrence on a tie, matching numpy's own `np.argmax` tie-breaking rule
/// - a real behavioral detail to match, not assume.
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
