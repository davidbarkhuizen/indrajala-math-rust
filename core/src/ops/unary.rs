//! Elementwise functions over one array: `exp` (correctly rounded), ReLU, softmax and the masks
//! the backward passes apply, `sum_axis0` and `argmax`.

use crate::array::{Array, Shape};
use crate::num::scalar::{self, max};
use crate::Error;

/// Elementwise `e^x` over a whole array - mirrors `array_layer.sigmoid`'s own reliance on
/// `exp`'s overflow behavior (a large negative `z` drives `exp(-z)` to `f64::INFINITY`, and
/// `1.0 / (1.0 + f64::INFINITY) == 0.0` under IEEE 754) instead of `math.exp`'s
/// `OverflowError`-raising behavior in the pure-Python reference - the one operation with a
/// documented overflow-boundary trap, checked here directly against `exp` before `sigmoid`
/// itself is ever built on top of it.
pub fn exp(arr: &Array) -> Array {
    let mut data = vec![0.0; arr.data.len()];
    crate::num::exp::exp_into(&arr.data, &mut data);
    Array { data, shape: arr.shape }
}

/// `exp` one value at a time (`num::exp::exp`, no vector lanes): the vector kernel's reference in
/// the tests, which check the two agree bit for bit.
pub fn exp_scalar(arr: &Array) -> Array {
    Array {
        data: arr.data.iter().map(|&v| crate::num::exp::exp(v)).collect(),
        shape: arr.shape,
    }
}

/// `exp` in place on a copy of `arr` (`num::exp::exp_in_place`, where `exp` writes a new buffer), and
/// how many values it redid one at a time: lanes whose rounding test failed or whose input is
/// outside the fast path's ordinary range, and the tail short of a vector. For the tests.
pub fn exp_lanes(arr: &Array) -> (Array, usize) {
    let mut data = arr.data.clone();
    let redone = crate::num::exp::exp_in_place(&mut data);
    (Array { data, shape: arr.shape }, redone)
}

/// Elementwise `erf` ([`crate::num::erf::erf`], fdlibm's, through the correctly rounded `exp`): GELU's
/// `erf(z / sqrt 2)`.
pub fn erf(arr: &Array) -> Array {
    Array {
        data: arr.data.iter().map(|&v| crate::num::erf::erf(v)).collect(),
        shape: arr.shape,
    }
}

/// Sums a 2D array's rows into a 1D vector - the one fixed-axis (`axis=0`) reduction this core
/// needs (`accumulate_gradient_batch`'s batched bias gradient,
/// `self._grad_b += self.delta_batch.sum(axis=0)`), cross-checked directly against that real
/// implementation. General axis-parameterized reduction stays out of scope; this is the one
/// fixed case, not a general `axis=` parameter.
pub fn sum_axis0(arr: &Array) -> Result<Array, Error> {
    match arr.shape {
        Shape::Matrix(rows, cols) => {
            let mut out = vec![0.0; cols];
            for row in 0..rows {
                for col in 0..cols {
                    out[col] += arr.data[row * cols + col];
                }
            }
            Ok(Array::from_vector(out))
        }
        Shape::Vector(_) => Err(Error::invalid("sum_axis0 requires a 2D array, got a 1D vector")),
    }
}

/// Elementwise `max(0.0, x)` - the Rust core's counterpart to `np.maximum(0.0, z)`
/// (`ArrayLayer.forward`'s array-level `relu_activation`). `np.maximum`/`np.where` remain
/// otherwise out of scope for this core; this is the one exception.
pub fn relu(arr: &Array) -> Array {
    Array {
        data: arr.data.iter().map(|&v| scalar::relu(v)).collect(),
        shape: arr.shape,
    }
}

/// Numerically-stable softmax normalization - `SoftmaxArrayLayer.forward`/`forward_batch`'s own
/// max-shift-then-exponentiate-then-normalize formula, validated directly against numpy
/// independently of either array-based softmax class, the same way `relu`/`mask::activation`
/// are validated for ReLU. A 1D vector is normalized as one whole distribution (`forward`'s
/// shape); a 2D matrix is normalized row-wise, one independent distribution per row
/// (`forward_batch`'s shape) - general axis-parameterized reduction stays out of scope; this is
/// the one fixed case, matching `sum_axis0`'s own precedent. Each row's sum is a left fold from
/// `0.0` in column order, numpy's `np.cumsum(e, axis=-1)[..., -1:]` (attention's and the token-wise
/// output's sum in indrajala-ml), not `e.sum()`'s pairwise sum.
pub fn softmax(arr: &Array) -> Array {
    fn normalize_row(row: &[f64]) -> Vec<f64> {
        let max_value = row.iter().fold(f64::NEG_INFINITY, |running, &v| max(running, v));
        let mut exp_values: Vec<f64> = row.iter().map(|&v| v - max_value).collect();
        crate::num::exp::exp_in_place(&mut exp_values);
        let total = exp_values.iter().fold(0.0, |acc, &e| acc + e);
        exp_values.iter().map(|&e| e / total).collect()
    }

    match arr.shape {
        Shape::Vector(_) => Array {
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
            Array {
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
pub fn argmax(arr: &Array) -> Result<usize, Error> {
    match arr.shape {
        Shape::Vector(n) => {
            if n == 0 {
                return Err(Error::invalid("argmax of an empty array"));
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
        Shape::Matrix(_, _) => Err(Error::invalid("argmax requires a 1D array")),
    }
}
