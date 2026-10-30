//! The bindings of the core's `ufuncs` (`indrajala_math::ufuncs`): each `#[pyfunction]` lends its
//! arrays to the core's and moves the results in.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::linalg::py_err;
#[allow(unused_imports)]
pub(crate) use indrajala_math::ufuncs::*;

/// Elementwise `e^x` over a whole array - mirrors `array_layer.sigmoid`'s own reliance on
/// `exp`'s overflow behavior (a large negative `z` drives `exp(-z)` to `f64::INFINITY`, and
/// `1.0 / (1.0 + f64::INFINITY) == 0.0` under IEEE 754) instead of `math.exp`'s
/// `OverflowError`-raising behavior in the pure-Python reference - the one operation with a
/// documented overflow-boundary trap, checked here directly against `exp` before `sigmoid`
/// itself is ever built on top of it.
#[pyfunction]
pub fn exp(arr: &RustArray) -> RustArray {
    RustArray::from(indrajala_math::ufuncs::exp(arr))
}

/// `exp` one value at a time (`cr_exp::exp`, no vector lanes): the vector kernel's reference in
/// the tests, which check the two agree bit for bit.
#[pyfunction]
pub fn exp_scalar(arr: &RustArray) -> RustArray {
    RustArray::from(indrajala_math::ufuncs::exp_scalar(arr))
}

/// `exp` in place on a copy of `arr` (`cr_exp::exp_in_place`, where `exp` writes a new buffer), and
/// how many values it redid one at a time: lanes whose rounding test failed or whose input is
/// outside the fast path's ordinary range, and the tail short of a vector. For the tests.
#[pyfunction]
pub fn exp_lanes(arr: &RustArray) -> (RustArray, usize) {
    {
        let (r0, r1) = indrajala_math::ufuncs::exp_lanes(arr);
        (RustArray::from(r0), r1)
    }
}

/// Sums a 2D array's rows into a 1D vector - the one fixed-axis (`axis=0`) reduction this core
/// needs (`accumulate_gradient_batch`'s batched bias gradient,
/// `self._grad_b += self.delta_batch.sum(axis=0)`), cross-checked directly against that real
/// implementation. General axis-parameterized reduction stays out of scope; this is the one
/// fixed case, not a general `axis=` parameter.
#[pyfunction]
pub fn sum_axis0(arr: &RustArray) -> PyResult<RustArray> {
    indrajala_math::ufuncs::sum_axis0(arr)
        .map(RustArray::from)
        .map_err(py_err)
}

/// Elementwise `max(0.0, x)` - the Rust core's counterpart to `np.maximum(0.0, z)`
/// (`ArrayLayer.forward`'s array-level `relu_activation`). `np.maximum`/`np.where` remain
/// otherwise out of scope for this core; this is the one exception.
#[pyfunction]
pub fn array_relu(arr: &RustArray) -> RustArray {
    RustArray::from(indrajala_math::ufuncs::array_relu(arr))
}

/// `downstream * (a > 0.0)` - ReLU's derivative mask, zeroing out the downstream gradient
/// wherever this layer's own cached activation was <= 0 (`ReLUArrayLayer.compute_hidden_delta`/
/// `compute_hidden_delta_batch`'s shared formula). Shape-agnostic (single-example 1D or batched
/// 2D), so one function covers both call shapes.
#[pyfunction]
pub fn array_relu_mask(downstream: &RustArray, a: &RustArray) -> PyResult<RustArray> {
    indrajala_math::ufuncs::array_relu_mask(downstream, a)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `downstream * a * (1 - a)` - the sigmoid's derivative applied to a downstream that a dense
/// layer's fused `layer_hidden_delta*` can't compute, because the next layer has no `W` to read
/// (a `LayerNorm`, or a residual fork whose body starts with one: indrajala-ml's layer-norm and
/// attention workplan, D5). The fused ops' elementwise expression, in their grouping, so given the
/// same downstream it computes their bits. Shape-agnostic, as `array_relu_mask`.
#[pyfunction]
pub fn array_sigmoid_mask(downstream: &RustArray, a: &RustArray) -> PyResult<RustArray> {
    indrajala_math::ufuncs::array_sigmoid_mask(downstream, a)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `downstream * base_activation * (1 - base_activation) * scale`, `scale = mask /
/// keep_probability` if `was_training` else `1.0` - the dropout layer's hidden delta from a
/// downstream it was handed, as `array_sigmoid_mask` is the sigmoid layer's (D5): the fused
/// `layer_dropout_hidden_delta*` ops' own elementwise expression. Shape-agnostic.
#[pyfunction]
pub fn array_dropout_mask(
    downstream: &RustArray,
    base_activation: &RustArray,
    mask: &RustArray,
    keep_probability: f64,
    was_training: bool,
) -> PyResult<RustArray> {
    indrajala_math::ufuncs::array_dropout_mask(downstream, base_activation, mask, keep_probability, was_training)
        .map(RustArray::from)
        .map_err(py_err)
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
#[pyfunction]
pub fn array_softmax(arr: &RustArray) -> RustArray {
    RustArray::from(indrajala_math::ufuncs::array_softmax(arr))
}

/// The index of the largest element in a 1D array - `classify_state`'s own
/// `np.argmax(self.predict_probabilities(state))`. Strict `>` (not `>=`) when scanning left to
/// right keeps the first occurrence on a tie, matching numpy's own `np.argmax` tie-breaking rule
/// - a real behavioral detail to match, not assume.
#[pyfunction]
pub fn argmax(arr: &RustArray) -> PyResult<usize> {
    indrajala_math::ufuncs::argmax(arr).map_err(py_err)
}
