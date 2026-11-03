//! The bindings of the core's `nn::dense`: each `#[pyfunction]` lends its arrays to the core's and
//! moves the results in.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::error::py_err;
use crate::num::activation_named;

/// `ArrayLayer.forward`: `sigmoid(self.W @ x + self.b)`, `x`/`b` both 1D.
#[pyfunction]
pub fn layer_forward(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::nn::dense::forward(w, x, b)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ArrayLayer.forward_batch`: `sigmoid(X @ self.W.T + self.b)`, `X` 2D (`batch, input_size`).
#[pyfunction]
pub fn layer_forward_batch(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::nn::dense::forward_batch(w, x, b)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ArrayLayer.compute_output_delta`/`compute_output_delta_batch`: `(a - reference) * a * (1 -
/// a)` - one elementwise formula, shape-agnostic (works for both the single-example 1D case and
/// the batched 2D case), so unlike `forward`/`forward_batch` this needs only one function.
#[pyfunction]
pub fn layer_output_delta(a: &RustArray, reference: &RustArray) -> PyResult<RustArray> {
    indrajala_math::nn::dense::output_delta(a, reference)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ArrayLayer.downstream`: `self.W.T @ self.delta`, the gradient a dense layer sends back to its
/// input. Unlike `hidden_downstream` below it has no upstream activation to check against, so a
/// conv or pool layer (which has no dense `W` of its own) can read it from the layer after it.
///
/// Computed as `delta @ W`, the same product, so `W` is read row by row and never copied into a transpose (the copy was
/// most of the call: 273 of 327 µs at 32 x 5408). It sums sequentially over `W`'s rows (`matmul`'s tiled kernel, as a
/// one-row product), not with `dot_product`'s 4-lane grouping, so the result differs from `W.T @ delta` by a few ULPs,
/// but is still the same on the scalar and AVX2 paths.
#[pyfunction]
pub fn layer_downstream(w: &RustArray, delta: &RustArray) -> PyResult<RustArray> {
    indrajala_math::nn::dense::downstream(w, delta)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ArrayLayer.downstream_batch`: `self.delta_batch @ self.W`.
#[pyfunction]
pub fn layer_downstream_batch(w: &RustArray, delta_batch: &RustArray) -> PyResult<RustArray> {
    indrajala_math::nn::dense::downstream_batch(w, delta_batch)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ArrayLayer.compute_hidden_delta`: `(next_layer.W.T @ next_layer.delta) * self.a * (1 -
/// self.a)`, single-example (`next_delta`/`a` both 1D).
#[pyfunction]
pub fn layer_hidden_delta(next_w: &RustArray, next_delta: &RustArray, a: &RustArray) -> PyResult<RustArray> {
    indrajala_math::nn::dense::hidden_delta(next_w, next_delta, a)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ArrayLayer.compute_hidden_delta_batch`: `(next_layer.delta_batch @ next_layer.W) * self.A *
/// (1 - self.A)` - batched; see `hidden_downstream_batch` for why this needs its own downstream
/// computation rather than reusing `hidden_downstream`.
#[pyfunction]
pub fn layer_hidden_delta_batch(
    next_w: &RustArray,
    next_delta_batch: &RustArray,
    a_batch: &RustArray,
) -> PyResult<RustArray> {
    indrajala_math::nn::dense::hidden_delta_batch(next_w, next_delta_batch, a_batch)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ArrayLayer.accumulate_gradient`: `self._grad_W += outer(delta, input_activation); self._grad_b
/// += delta`, single-example. Returns the updated `(grad_W, grad_b)` pair rather than mutating in
/// place - `Array` is immutable from Python's own operators (`+=` rebinds via `__iadd__`, which
/// this crate already has), so the caller rebinds `self._grad_W`/`self._grad_b` the same way.
#[pyfunction]
pub fn layer_accumulate_gradient(
    delta: &RustArray,
    input_activation: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::nn::dense::accumulate_gradient(delta, input_activation, grad_w, grad_b)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

/// `ArrayLayer.accumulate_gradient_batch`: `self._grad_W += self.delta_batch.T @
/// input_activation_batch; self._grad_b += self.delta_batch.sum(axis=0)`.
#[pyfunction]
pub fn layer_accumulate_gradient_batch(
    delta_batch: &RustArray,
    input_activation_batch: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::nn::dense::accumulate_gradient_batch(delta_batch, input_activation_batch, grad_w, grad_b)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

/// `(a, z | None)` into Python.
fn with_preactivation(
    (a, z): (indrajala_math::Array, Option<indrajala_math::Array>),
) -> (RustArray, Option<RustArray>) {
    (RustArray::from(a), z.map(RustArray::from))
}

/// A hidden dense layer's forward with `activation` (`"relu"`, `"gelu"`, `"gelu_tanh"`):
/// `act(self.W @ x + self.b)`, `x`/`b` both 1D, in one call. Returns `(a, z)`, `z` the
/// pre-activation for the GELUs (their derivative reads it) and `None` for ReLU.
#[pyfunction]
pub fn layer_activation_forward(
    w: &RustArray,
    x: &RustArray,
    b: &RustArray,
    activation: &str,
) -> PyResult<(RustArray, Option<RustArray>)> {
    let activation = activation_named(activation)?;
    indrajala_math::nn::dense::activation_forward(w, x, b, activation)
        .map(with_preactivation)
        .map_err(py_err)
}

/// `layer_activation_forward`, batched: `act(X @ self.W.T + self.b)`, `X` 2D (`batch,
/// input_size`).
#[pyfunction]
pub fn layer_activation_forward_batch(
    w: &RustArray,
    x: &RustArray,
    b: &RustArray,
    activation: &str,
) -> PyResult<(RustArray, Option<RustArray>)> {
    let activation = activation_named(activation)?;
    indrajala_math::nn::dense::activation_forward_batch(w, x, b, activation)
        .map(with_preactivation)
        .map_err(py_err)
}

/// A hidden dense layer's delta: `(next_layer.W.T @ next_layer.delta) * act'(cached)`,
/// single-example; `cached` is what the forward kept, `a` for ReLU (its mask `a > 0`), `z` for the
/// GELUs.
#[pyfunction]
pub fn layer_activation_hidden_delta(
    next_w: &RustArray,
    next_delta: &RustArray,
    cached: &RustArray,
    activation: &str,
) -> PyResult<RustArray> {
    let activation = activation_named(activation)?;
    indrajala_math::nn::dense::activation_hidden_delta(next_w, next_delta, cached, activation)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `layer_activation_hidden_delta`, batched: `(next_layer.delta_batch @ next_layer.W) *
/// act'(cached_batch)`, no transpose on `next_w`.
#[pyfunction]
pub fn layer_activation_hidden_delta_batch(
    next_w: &RustArray,
    next_delta_batch: &RustArray,
    cached_batch: &RustArray,
    activation: &str,
) -> PyResult<RustArray> {
    let activation = activation_named(activation)?;
    indrajala_math::nn::dense::activation_hidden_delta_batch(next_w, next_delta_batch, cached_batch, activation)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `AffineArrayLayer.forward`: `self.W @ x + self.b` with no activation, the affine layer that ends
/// a residual block's body (indrajala-ml's residual-connections workplan, D4): `linear_preactivation`
/// itself, so its bits are the pre-activation every `layer_*forward` computes.
#[pyfunction]
pub fn affine_forward(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::nn::dense::affine_forward(w, x, b)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `AffineArrayLayer.forward_batch`: `X @ self.W.T + self.b`, `linear_preactivation_batch`, so row
/// `i` is bit-identical to `affine_forward(w, X[i], b)`.
#[pyfunction]
pub fn affine_forward_batch(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::nn::dense::affine_forward_batch(w, x, b)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ArrayLayer.compute_hidden_delta` when the next layer is a residual block's fork: `(body_w.T @
/// body_delta + skip) * self.a * (1 - self.a)`, `body_w`/`body_delta` the body's first layer's and
/// `skip` the block's add's delta - `layer_hidden_delta` with the skip term added to the downstream
/// before the sigmoid's derivative, in one call. Single-example (`body_delta`/`skip`/`a` all 1D).
#[pyfunction]
pub fn layer_hidden_delta_skip(
    body_w: &RustArray,
    body_delta: &RustArray,
    skip: &RustArray,
    a: &RustArray,
) -> PyResult<RustArray> {
    indrajala_math::nn::dense::hidden_delta_skip(body_w, body_delta, skip, a)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `layer_hidden_delta_skip`, batched: `(body_delta_batch @ body_w + skip_batch) * self.A * (1 -
/// self.A)`.
#[pyfunction]
pub fn layer_hidden_delta_skip_batch(
    body_w: &RustArray,
    body_delta_batch: &RustArray,
    skip_batch: &RustArray,
    a_batch: &RustArray,
) -> PyResult<RustArray> {
    indrajala_math::nn::dense::hidden_delta_skip_batch(body_w, body_delta_batch, skip_batch, a_batch)
        .map(RustArray::from)
        .map_err(py_err)
}

/// A hidden dense layer's delta when the next layer is a residual block's fork: `(body_w.T @
/// body_delta + skip) * act'(cached)`, `layer_activation_hidden_delta` with the skip term.
#[pyfunction]
pub fn layer_activation_hidden_delta_skip(
    body_w: &RustArray,
    body_delta: &RustArray,
    skip: &RustArray,
    cached: &RustArray,
    activation: &str,
) -> PyResult<RustArray> {
    let activation = activation_named(activation)?;
    indrajala_math::nn::dense::activation_hidden_delta_skip(body_w, body_delta, skip, cached, activation)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `layer_activation_hidden_delta_skip`, batched: `(body_delta_batch @ body_w + skip_batch) *
/// act'(cached_batch)`.
#[pyfunction]
pub fn layer_activation_hidden_delta_skip_batch(
    body_w: &RustArray,
    body_delta_batch: &RustArray,
    skip_batch: &RustArray,
    cached_batch: &RustArray,
    activation: &str,
) -> PyResult<RustArray> {
    let activation = activation_named(activation)?;
    indrajala_math::nn::dense::activation_hidden_delta_skip_batch(
        body_w,
        body_delta_batch,
        skip_batch,
        cached_batch,
        activation,
    )
    .map(RustArray::from)
    .map_err(py_err)
}

/// `LinearArrayLayer.forward`: `self.W @ x`, `x` 1D, `linear_preactivation` without the bias.
#[pyfunction]
pub fn linear_forward(w: &RustArray, x: &RustArray) -> PyResult<RustArray> {
    indrajala_math::nn::dense::linear_forward(w, x)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `LinearArrayLayer.forward_batch`: `X @ self.W.T`, `linear_preactivation_batch` without the
/// bias, so row `i` is bit-identical to `linear_forward(w, X[i])`.
#[pyfunction]
pub fn linear_forward_batch(w: &RustArray, x: &RustArray) -> PyResult<RustArray> {
    indrajala_math::nn::dense::linear_forward_batch(w, x)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `LinearArrayLayer.accumulate_gradient_batch`: `self.grad_W += self.delta_batch.T @
/// input_activation_batch`, `layer_accumulate_gradient_batch`'s `grad_W` without its `grad_b`.
#[pyfunction]
pub fn linear_accumulate_gradient_batch(
    delta_batch: &RustArray,
    input_activation_batch: &RustArray,
    grad_w: &RustArray,
) -> PyResult<RustArray> {
    indrajala_math::nn::dense::linear_accumulate_gradient_batch(delta_batch, input_activation_batch, grad_w)
        .map(RustArray::from)
        .map_err(py_err)
}

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(layer_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layer_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_output_delta, m)?)?;
    m.add_function(wrap_pyfunction!(layer_downstream, m)?)?;
    m.add_function(wrap_pyfunction!(layer_downstream_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_hidden_delta, m)?)?;
    m.add_function(wrap_pyfunction!(layer_hidden_delta_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_accumulate_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(layer_accumulate_gradient_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_activation_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layer_activation_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_activation_hidden_delta, m)?)?;
    m.add_function(wrap_pyfunction!(layer_activation_hidden_delta_batch, m)?)?;
    m.add_function(wrap_pyfunction!(affine_forward, m)?)?;
    m.add_function(wrap_pyfunction!(affine_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_hidden_delta_skip, m)?)?;
    m.add_function(wrap_pyfunction!(layer_hidden_delta_skip_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_activation_hidden_delta_skip, m)?)?;
    m.add_function(wrap_pyfunction!(layer_activation_hidden_delta_skip_batch, m)?)?;
    m.add_function(wrap_pyfunction!(linear_forward, m)?)?;
    m.add_function(wrap_pyfunction!(linear_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(linear_accumulate_gradient_batch, m)?)?;
    Ok(())
}
