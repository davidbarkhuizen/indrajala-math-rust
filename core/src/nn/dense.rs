//! One Rust function per `ArrayLayer` method, doing the entire computation in one call instead
//! of composing it from several separate `Array` operator/ufunc calls in Python - each of those
//! crosses the Python/Rust boundary and allocates a new `Array`, and call count (not per-call
//! cost) dominates this crate's matmul-bound cost. Every function here mirrors one
//! `indrajala_ml/model/layers/numpy/array_layer.py` method's formula exactly - see that file for the
//! reference this crate is checked against.

use crate::array::{Array, Shape};
use crate::linalg::{matmul, matmul_nt, matmul_transposed_add};
use crate::num::activation::Activation;
use crate::ops::arith::require_same_shape;
use crate::ops::arith::same_shape_elementwise;
use crate::ops::mask::masked;
use crate::ops::unary::sum_axis0;
use crate::Error;

pub(super) fn sigmoid(z: &Array) -> Array {
    let mut data = z.data.clone();
    crate::num::exp::sigmoid_in_place(&mut data);
    Array { data, shape: z.shape }
}

/// `self.W @ x + self.b`, `x`/`b` both 1D - the pre-activation shared by every `layer_*forward`
/// below (plain sigmoid, ReLU, softmax, dropout's own pre-mask sigmoid); each differs only in
/// what it does to this value, not in how it's computed.
pub(super) fn linear_preactivation(w: &Array, x: &Array, b: &Array) -> Result<Array, Error> {
    let z = matmul(w, x)?;
    z.combine_with_array(b, |a, bv| a + bv, "add")
}

/// `X @ self.W.T + self.b`, `X` 2D (`batch, input_size`) - the batched analogue of
/// `linear_preactivation` above, shared the same way by every `layer_*forward_batch`.
///
/// `matmul_nt` reads `W` row by row, so `W.T` is never copied (the copy was 266 of 326 µs at 32
/// x 5408, batch 1). Each row of `X @ W.T` is the same `dot_product` calls as `W @ x`, so row `i`
/// here is bit-identical to `linear_preactivation(w, X[i], b)`.
pub(super) fn linear_preactivation_batch(w: &Array, x: &Array, b: &Array) -> Result<Array, Error> {
    let z = matmul_nt(x, w)?;
    z.combine_with_array(b, |a, bv| a + bv, "add")
}

/// `ArrayLayer.forward`: `sigmoid(self.W @ x + self.b)`, `x`/`b` both 1D.
pub fn forward(w: &Array, x: &Array, b: &Array) -> Result<Array, Error> {
    Ok(sigmoid(&linear_preactivation(w, x, b)?))
}

/// `ArrayLayer.forward_batch`: `sigmoid(X @ self.W.T + self.b)`, `X` 2D (`batch, input_size`).
pub fn forward_batch(w: &Array, x: &Array, b: &Array) -> Result<Array, Error> {
    Ok(sigmoid(&linear_preactivation_batch(w, x, b)?))
}

/// `ArrayLayer.compute_output_delta`/`compute_output_delta_batch`: `(a - reference) * a * (1 -
/// a)` - one elementwise formula, shape-agnostic (works for both the single-example 1D case and
/// the batched 2D case), so unlike `forward`/`forward_batch` this needs only one function.
pub fn output_delta(a: &Array, reference: &Array) -> Result<Array, Error> {
    require_same_shape(a, reference, "layer_output_delta")?;
    let data = a
        .data
        .iter()
        .zip(reference.data.iter())
        .map(|(&av, &rv)| (av - rv) * av * (1.0 - av))
        .collect();
    Ok(Array { data, shape: a.shape })
}

/// `ArrayLayer.downstream`: `self.W.T @ self.delta`, the gradient a dense layer sends back to its
/// input. Unlike `hidden_downstream` below it has no upstream activation to check against, so a
/// conv or pool layer (which has no dense `W` of its own) can read it from the layer after it.
///
/// Computed as `delta @ W`, the same product, so `W` is read row by row and never copied into a transpose (the copy was
/// most of the call: 273 of 327 µs at 32 x 5408). It sums sequentially over `W`'s rows (`matmul`'s tiled kernel, as a
/// one-row product), not with `dot_product`'s 4-lane grouping, so the result differs from `W.T @ delta` by a few ULPs,
/// but is still the same on the scalar and AVX2 paths.
pub fn downstream(w: &Array, delta: &Array) -> Result<Array, Error> {
    if !matches!(delta.shape, Shape::Vector(_)) {
        return Err(Error::invalid(format!(
            "layer_downstream requires a 1D delta, got shape {:?}",
            delta.shape
        )));
    }
    matmul(delta, w)
}

/// `ArrayLayer.downstream_batch`: `self.delta_batch @ self.W`.
pub fn downstream_batch(w: &Array, delta_batch: &Array) -> Result<Array, Error> {
    matmul(delta_batch, w)
}

/// `next_layer.W.T @ next_layer.delta`, single-example (`next_delta`/`a` both 1D), shape-checked
/// against `a` - the downstream term shared by every `layer_*hidden_delta` below (plain sigmoid,
/// ReLU, dropout); each differs only in what elementwise formula it applies on top, not in how
/// the downstream matmul itself is computed.
pub(super) fn hidden_downstream(next_w: &Array, next_delta: &Array, a: &Array, context: &str) -> Result<Array, Error> {
    let downstream = downstream(next_w, next_delta)?;
    require_same_shape(&downstream, a, context)?;
    Ok(downstream)
}

/// `next_layer.delta_batch @ next_layer.W` - the batched analogue of `hidden_downstream` above,
/// no transpose on `next_w` (`next_delta_batch`'s batch axis is on the left instead of `next_w`'s
/// being on the left, a genuinely different call shape, not just a shape-agnostic reuse).
pub(super) fn hidden_downstream_batch(
    next_w: &Array,
    next_delta_batch: &Array,
    a_batch: &Array,
    context: &str,
) -> Result<Array, Error> {
    let downstream = downstream_batch(next_w, next_delta_batch)?;
    require_same_shape(&downstream, a_batch, context)?;
    Ok(downstream)
}

/// `ArrayLayer.compute_hidden_delta`: `(next_layer.W.T @ next_layer.delta) * self.a * (1 -
/// self.a)`, single-example (`next_delta`/`a` both 1D).
pub fn hidden_delta(next_w: &Array, next_delta: &Array, a: &Array) -> Result<Array, Error> {
    let downstream = hidden_downstream(next_w, next_delta, a, "layer_hidden_delta")?;
    let data = downstream
        .data
        .iter()
        .zip(a.data.iter())
        .map(|(&d, &av)| d * av * (1.0 - av))
        .collect();
    Ok(Array { data, shape: a.shape })
}

/// `ArrayLayer.compute_hidden_delta_batch`: `(next_layer.delta_batch @ next_layer.W) * self.A *
/// (1 - self.A)` - batched; see `hidden_downstream_batch` for why this needs its own downstream
/// computation rather than reusing `hidden_downstream`.
pub fn hidden_delta_batch(next_w: &Array, next_delta_batch: &Array, a_batch: &Array) -> Result<Array, Error> {
    let downstream = hidden_downstream_batch(next_w, next_delta_batch, a_batch, "layer_hidden_delta_batch")?;
    let data = downstream
        .data
        .iter()
        .zip(a_batch.data.iter())
        .map(|(&d, &av)| d * av * (1.0 - av))
        .collect();
    Ok(Array {
        data,
        shape: a_batch.shape,
    })
}

/// `ArrayLayer.accumulate_gradient`: `self._grad_W += outer(delta, input_activation); self._grad_b
/// += delta`, single-example. Returns the updated `(grad_W, grad_b)` pair rather than mutating in
/// place - `Array` is immutable from Python's own operators (`+=` rebinds via `__iadd__`, which
/// this crate already has), so the caller rebinds `self._grad_W`/`self._grad_b` the same way.
pub fn accumulate_gradient(
    delta: &Array,
    input_activation: &Array,
    grad_w: &Array,
    grad_b: &Array,
) -> Result<(Array, Array), Error> {
    let (m, n) = match (delta.shape, input_activation.shape) {
        (Shape::Vector(m), Shape::Vector(n)) => (m, n),
        (d_shape, x_shape) => {
            return Err(Error::invalid(format!(
                "layer_accumulate_gradient requires 1D delta and input_activation, got shapes {:?} and {:?}",
                d_shape, x_shape
            )))
        }
    };
    if grad_w.shape != Shape::Matrix(m, n) {
        return Err(Error::invalid(format!(
            "layer_accumulate_gradient requires grad_w of shape {:?}, got {:?}",
            Shape::Matrix(m, n),
            grad_w.shape
        )));
    }
    // One pass and one allocation, instead of building outer(delta, x) and then adding it. A
    // plain multiply then add, not mul_add: two roundings, the same as the product and the sum
    // done separately, so the result is bit-identical to grad_w + outer(delta, x). LLVM
    // vectorizes the inner loop as separate multiplies and adds, which keeps both roundings.
    let x = &input_activation.data;
    let mut data = Vec::with_capacity(m * n);
    for (grad_w_row, &d) in grad_w.data.chunks_exact(n.max(1)).zip(delta.data.iter()) {
        data.extend(grad_w_row.iter().zip(x.iter()).map(|(&g, &xj)| g + d * xj));
    }
    let new_grad_w = Array::from_matrix(data, m, n);
    let new_grad_b = grad_b.combine_with_array(delta, |g, d| g + d, "add")?;
    Ok((new_grad_w, new_grad_b))
}

/// `ArrayLayer.accumulate_gradient_batch`: `self._grad_W += self.delta_batch.T @
/// input_activation_batch; self._grad_b += self.delta_batch.sum(axis=0)`.
pub fn accumulate_gradient_batch(
    delta_batch: &Array,
    input_activation_batch: &Array,
    grad_w: &Array,
    grad_b: &Array,
) -> Result<(Array, Array), Error> {
    // grad_w + delta_batch.T @ X in the product's own store, bit-identical to the separate add
    let new_grad_w = matmul_transposed_add(grad_w, delta_batch, input_activation_batch)?;
    let grad_b_update = sum_axis0(delta_batch)?;
    let new_grad_b = grad_b.combine_with_array(&grad_b_update, |g, u| g + u, "add")?;
    Ok((new_grad_w, new_grad_b))
}

/// The forward of a hidden dense layer with `activation` (`ReLUArrayLayer.forward` and its GELU
/// twins): `act(self.W @ x + self.b)`, `x`/`b` both 1D, in one call. Returns `(a, z)`, `z` the
/// pre-activation when the activation's derivative reads it
/// ([`Activation::keeps_preactivation`], the GELUs), else `None` (ReLU, whose arithmetic is
/// `max(0, z)` by [`crate::num::scalar::relu`] as it always was).
pub fn activation_forward(
    w: &Array,
    x: &Array,
    b: &Array,
    activation: Activation,
) -> Result<(Array, Option<Array>), Error> {
    Ok(activated(linear_preactivation(w, x, b)?, activation))
}

/// `activation_forward`, batched: `act(X @ self.W.T + self.b)`, `X` 2D (`batch,
/// input_size`).
pub fn activation_forward_batch(
    w: &Array,
    x: &Array,
    b: &Array,
    activation: Activation,
) -> Result<(Array, Option<Array>), Error> {
    Ok(activated(linear_preactivation_batch(w, x, b)?, activation))
}

/// `(act(z), z)` if the activation keeps `z`, else `(act(z), None)`, moving `z` rather than
/// copying it.
fn activated(z: Array, activation: Activation) -> (Array, Option<Array>) {
    let a = Array {
        data: activation.apply(&z.data),
        shape: z.shape,
    };
    if activation.keeps_preactivation() {
        (a, Some(z))
    } else {
        (a, None)
    }
}

/// The hidden delta of a dense layer with `activation`: `(next_layer.W.T @ next_layer.delta) *
/// act'(cached)`, single-example (`next_delta`/`cached` both 1D); `cached` is what
/// `activation_forward` kept, `a` for ReLU (its mask `a > 0`), `z` for the GELUs.
pub fn activation_hidden_delta(
    next_w: &Array,
    next_delta: &Array,
    cached: &Array,
    activation: Activation,
) -> Result<Array, Error> {
    let context = "layer_activation_hidden_delta";
    let downstream = hidden_downstream(next_w, next_delta, cached, context)?;
    Ok(masked(&downstream, cached, activation))
}

/// `activation_hidden_delta`, batched: `(next_layer.delta_batch @ next_layer.W) *
/// act'(cached_batch)`, no transpose on `next_w` (the shape-of-call-sites distinction
/// `hidden_delta_batch` documents).
pub fn activation_hidden_delta_batch(
    next_w: &Array,
    next_delta_batch: &Array,
    cached_batch: &Array,
    activation: Activation,
) -> Result<Array, Error> {
    let context = "layer_activation_hidden_delta_batch";
    let downstream = hidden_downstream_batch(next_w, next_delta_batch, cached_batch, context)?;
    Ok(masked(&downstream, cached_batch, activation))
}

/// `AffineArrayLayer.forward`: `self.W @ x + self.b` with no activation, the affine layer that ends
/// a residual block's body (indrajala-ml's residual-connections workplan, D4): `linear_preactivation`
/// itself, so its bits are the pre-activation every `layer_*forward` computes.
pub fn affine_forward(w: &Array, x: &Array, b: &Array) -> Result<Array, Error> {
    linear_preactivation(w, x, b)
}

/// `AffineArrayLayer.forward_batch`: `X @ self.W.T + self.b`, `linear_preactivation_batch`, so row
/// `i` is bit-identical to `affine_forward(w, X[i], b)`.
pub fn affine_forward_batch(w: &Array, x: &Array, b: &Array) -> Result<Array, Error> {
    linear_preactivation_batch(w, x, b)
}

/// `downstream + skip`, elementwise, shape-checked against `skip`: the downstream of a residual
/// block's fork (its body's first layer's, `downstream`) plus the identity path's (its add's delta,
/// `skip`), shared by every `layer_*hidden_delta_skip` below. One IEEE addition per element, so the
/// operand order can't move bits.
pub(super) fn with_skip(downstream: Array, skip: &Array, context: &str) -> Result<Array, Error> {
    require_same_shape(&downstream, skip, context)?;
    Ok(Array {
        data: same_shape_elementwise(&downstream.data, &skip.data, |d, s| d + s),
        shape: downstream.shape,
    })
}

/// `ArrayLayer.compute_hidden_delta` when the next layer is a residual block's fork: `(body_w.T @
/// body_delta + skip) * self.a * (1 - self.a)`, `body_w`/`body_delta` the body's first layer's and
/// `skip` the block's add's delta - `hidden_delta` with the skip term added to the downstream
/// before the sigmoid's derivative, in one call. Single-example (`body_delta`/`skip`/`a` all 1D).
pub fn hidden_delta_skip(body_w: &Array, body_delta: &Array, skip: &Array, a: &Array) -> Result<Array, Error> {
    let downstream = hidden_downstream(body_w, body_delta, a, "layer_hidden_delta_skip")?;
    let downstream = with_skip(downstream, skip, "layer_hidden_delta_skip")?;
    let data = same_shape_elementwise(&downstream.data, &a.data, |d, av| d * av * (1.0 - av));
    Ok(Array { data, shape: a.shape })
}

/// `hidden_delta_skip`, batched: `(body_delta_batch @ body_w + skip_batch) * self.A * (1 -
/// self.A)`.
pub fn hidden_delta_skip_batch(
    body_w: &Array,
    body_delta_batch: &Array,
    skip_batch: &Array,
    a_batch: &Array,
) -> Result<Array, Error> {
    let downstream = hidden_downstream_batch(body_w, body_delta_batch, a_batch, "layer_hidden_delta_skip_batch")?;
    let downstream = with_skip(downstream, skip_batch, "layer_hidden_delta_skip_batch")?;
    let data = same_shape_elementwise(&downstream.data, &a_batch.data, |d, av| d * av * (1.0 - av));
    Ok(Array {
        data,
        shape: a_batch.shape,
    })
}

/// The hidden delta of a dense layer with `activation` when the next layer is a residual block's
/// fork: `(body_w.T @ body_delta + skip) * act'(cached)`, `activation_hidden_delta` with the
/// skip term.
pub fn activation_hidden_delta_skip(
    body_w: &Array,
    body_delta: &Array,
    skip: &Array,
    cached: &Array,
    activation: Activation,
) -> Result<Array, Error> {
    let context = "layer_activation_hidden_delta_skip";
    let downstream = hidden_downstream(body_w, body_delta, cached, context)?;
    let downstream = with_skip(downstream, skip, context)?;
    Ok(masked(&downstream, cached, activation))
}

/// `activation_hidden_delta_skip`, batched: `(body_delta_batch @ body_w + skip_batch) *
/// act'(cached_batch)`.
pub fn activation_hidden_delta_skip_batch(
    body_w: &Array,
    body_delta_batch: &Array,
    skip_batch: &Array,
    cached_batch: &Array,
    activation: Activation,
) -> Result<Array, Error> {
    let context = "layer_activation_hidden_delta_skip_batch";
    let downstream = hidden_downstream_batch(body_w, body_delta_batch, cached_batch, context)?;
    let downstream = with_skip(downstream, skip_batch, context)?;
    Ok(masked(&downstream, cached_batch, activation))
}

/// `LinearArrayLayer.forward`: `self.W @ x`, `x` 1D, `linear_preactivation` without the bias.
pub fn linear_forward(w: &Array, x: &Array) -> Result<Array, Error> {
    if !matches!(x.shape, Shape::Vector(_)) {
        return Err(Error::invalid(format!(
            "linear_forward requires a 1D x, got shape {:?}",
            x.shape
        )));
    }
    matmul(w, x)
}

/// `LinearArrayLayer.forward_batch`: `X @ self.W.T`, `linear_preactivation_batch` without the
/// bias, so row `i` is bit-identical to `linear_forward(w, X[i])`.
pub fn linear_forward_batch(w: &Array, x: &Array) -> Result<Array, Error> {
    matmul_nt(x, w)
}

/// `LinearArrayLayer.accumulate_gradient_batch`: `self.grad_W += self.delta_batch.T @
/// input_activation_batch`, `accumulate_gradient_batch`'s `grad_W` without its `grad_b`.
pub fn linear_accumulate_gradient_batch(
    delta_batch: &Array,
    input_activation_batch: &Array,
    grad_w: &Array,
) -> Result<Array, Error> {
    matmul_transposed_add(grad_w, delta_batch, input_activation_batch)
}
