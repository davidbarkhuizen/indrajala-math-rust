use crate::array::{Array, Shape};
use crate::fused::dropout_hidden_delta_from_downstream;
use crate::ops::same_shape_elementwise;
use crate::scalar::{max, relu};
use crate::Error;

/// Elementwise `e^x` over a whole array - mirrors `array_layer.sigmoid`'s own reliance on
/// `exp`'s overflow behavior (a large negative `z` drives `exp(-z)` to `f64::INFINITY`, and
/// `1.0 / (1.0 + f64::INFINITY) == 0.0` under IEEE 754) instead of `math.exp`'s
/// `OverflowError`-raising behavior in the pure-Python reference - the one operation with a
/// documented overflow-boundary trap, checked here directly against `exp` before `sigmoid`
/// itself is ever built on top of it.
pub fn exp(arr: &Array) -> Array {
    let mut data = vec![0.0; arr.data.len()];
    crate::cr_exp::exp_into(&arr.data, &mut data);
    Array { data, shape: arr.shape }
}

/// `exp` one value at a time (`cr_exp::exp`, no vector lanes): the vector kernel's reference in
/// the tests, which check the two agree bit for bit.
pub fn exp_scalar(arr: &Array) -> Array {
    Array {
        data: arr.data.iter().map(|&v| crate::cr_exp::exp(v)).collect(),
        shape: arr.shape,
    }
}

/// `exp` in place on a copy of `arr` (`cr_exp::exp_in_place`, where `exp` writes a new buffer), and
/// how many values it redid one at a time: lanes whose rounding test failed or whose input is
/// outside the fast path's ordinary range, and the tail short of a vector. For the tests.
pub fn exp_lanes(arr: &Array) -> (Array, usize) {
    let mut data = arr.data.clone();
    let redone = crate::cr_exp::exp_in_place(&mut data);
    (Array { data, shape: arr.shape }, redone)
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
pub fn array_relu(arr: &Array) -> Array {
    Array {
        data: arr.data.iter().map(|&v| relu(v)).collect(),
        shape: arr.shape,
    }
}

/// `downstream * (a > 0.0)` - ReLU's derivative mask, zeroing out the downstream gradient
/// wherever this layer's own cached activation was <= 0 (`ReLUArrayLayer.compute_hidden_delta`/
/// `compute_hidden_delta_batch`'s shared formula). Shape-agnostic (single-example 1D or batched
/// 2D), so one function covers both call shapes.
pub fn array_relu_mask(downstream: &Array, a: &Array) -> Result<Array, Error> {
    if downstream.shape != a.shape {
        return Err(Error::invalid(format!(
            "array_relu_mask requires matching shapes, got {:?} and {:?}",
            downstream.shape, a.shape
        )));
    }
    Ok(Array {
        data: same_shape_elementwise(&downstream.data, &a.data, |d, av| if av > 0.0 { d } else { 0.0 }),
        shape: downstream.shape,
    })
}

/// `downstream * a * (1 - a)` - the sigmoid's derivative applied to a downstream that a dense
/// layer's fused `layer_hidden_delta*` can't compute, because the next layer has no `W` to read
/// (a `LayerNorm`, or a residual fork whose body starts with one: indrajala-ml's layer-norm and
/// attention workplan, D5). The fused ops' elementwise expression, in their grouping, so given the
/// same downstream it computes their bits. Shape-agnostic, as `array_relu_mask`.
pub fn array_sigmoid_mask(downstream: &Array, a: &Array) -> Result<Array, Error> {
    if downstream.shape != a.shape {
        return Err(Error::invalid(format!(
            "array_sigmoid_mask requires matching shapes, got {:?} and {:?}",
            downstream.shape, a.shape
        )));
    }
    Ok(Array {
        data: same_shape_elementwise(&downstream.data, &a.data, |d, av| d * av * (1.0 - av)),
        shape: downstream.shape,
    })
}

/// `downstream * base_activation * (1 - base_activation) * scale`, `scale = mask /
/// keep_probability` if `was_training` else `1.0` - the dropout layer's hidden delta from a
/// downstream it was handed, as `array_sigmoid_mask` is the sigmoid layer's (D5): the fused
/// `layer_dropout_hidden_delta*` ops' own elementwise expression. Shape-agnostic.
pub fn array_dropout_mask(
    downstream: &Array,
    base_activation: &Array,
    mask: &Array,
    keep_probability: f64,
    was_training: bool,
) -> Result<Array, Error> {
    if downstream.shape != base_activation.shape || mask.shape != base_activation.shape {
        return Err(Error::invalid(format!(
            "array_dropout_mask requires matching shapes, got {:?}, {:?} and {:?}",
            downstream.shape, base_activation.shape, mask.shape
        )));
    }
    Ok(dropout_hidden_delta_from_downstream(
        downstream,
        base_activation,
        mask,
        keep_probability,
        was_training,
    ))
}

/// Numerically-stable softmax normalization - `SoftmaxArrayLayer.forward`/`forward_batch`'s own
/// max-shift-then-exponentiate-then-normalize formula, validated directly against numpy
/// independently of either array-based softmax class, the same way `array_relu`/`array_relu_mask`
/// are validated for ReLU. A 1D vector is normalized as one whole distribution (`forward`'s
/// shape); a 2D matrix is normalized row-wise, one independent distribution per row
/// (`forward_batch`'s shape) - general axis-parameterized reduction stays out of scope; this is
/// the one fixed case, matching `sum_axis0`'s own precedent. Each row's sum is a left fold from
/// `0.0` in column order, numpy's `np.cumsum(e, axis=-1)[..., -1:]` (attention's and the token-wise
/// output's sum in indrajala-ml), not `e.sum()`'s pairwise sum.
pub fn array_softmax(arr: &Array) -> Array {
    fn normalize_row(row: &[f64]) -> Vec<f64> {
        let max_value = row.iter().fold(f64::NEG_INFINITY, |running, &v| max(running, v));
        let mut exp_values: Vec<f64> = row.iter().map(|&v| v - max_value).collect();
        crate::cr_exp::exp_in_place(&mut exp_values);
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
