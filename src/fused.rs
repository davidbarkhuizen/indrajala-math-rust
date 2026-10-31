//! The bindings of the core's `fused` (`indrajala_math::fused`): each `#[pyfunction]` lends its
//! arrays to the core's and moves the results in.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::generator::Generator;
use crate::linalg::py_err;
#[allow(unused_imports)]
pub(crate) use indrajala_math::fused::*;

/// `ArrayLayer.forward`: `sigmoid(self.W @ x + self.b)`, `x`/`b` both 1D.
#[pyfunction]
pub fn layer_forward(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::fused::layer_forward(w, x, b)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ArrayLayer.forward_batch`: `sigmoid(X @ self.W.T + self.b)`, `X` 2D (`batch, input_size`).
#[pyfunction]
pub fn layer_forward_batch(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::fused::layer_forward_batch(w, x, b)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ArrayLayer.compute_output_delta`/`compute_output_delta_batch`: `(a - reference) * a * (1 -
/// a)` - one elementwise formula, shape-agnostic (works for both the single-example 1D case and
/// the batched 2D case), so unlike `forward`/`forward_batch` this needs only one function.
#[pyfunction]
pub fn layer_output_delta(a: &RustArray, reference: &RustArray) -> PyResult<RustArray> {
    indrajala_math::fused::layer_output_delta(a, reference)
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
    indrajala_math::fused::layer_downstream(w, delta)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ArrayLayer.downstream_batch`: `self.delta_batch @ self.W`.
#[pyfunction]
pub fn layer_downstream_batch(w: &RustArray, delta_batch: &RustArray) -> PyResult<RustArray> {
    indrajala_math::fused::layer_downstream_batch(w, delta_batch)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ArrayLayer.compute_hidden_delta`: `(next_layer.W.T @ next_layer.delta) * self.a * (1 -
/// self.a)`, single-example (`next_delta`/`a` both 1D).
#[pyfunction]
pub fn layer_hidden_delta(next_w: &RustArray, next_delta: &RustArray, a: &RustArray) -> PyResult<RustArray> {
    indrajala_math::fused::layer_hidden_delta(next_w, next_delta, a)
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
    indrajala_math::fused::layer_hidden_delta_batch(next_w, next_delta_batch, a_batch)
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
    indrajala_math::fused::layer_accumulate_gradient(delta, input_activation, grad_w, grad_b)
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
    indrajala_math::fused::layer_accumulate_gradient_batch(delta_batch, input_activation_batch, grad_w, grad_b)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

/// `ArrayLayer.apply_accumulated_gradient`: minibatch SGD in Goyal et al. 2017's grouping, eq. (2),
/// the mean gradient first and then the rate: `self.W -= learning_rate * (self._grad_W /
/// batch_size); self.b -= learning_rate * (self._grad_b / batch_size)` - shape-agnostic
/// (`W`/`grad_W` are always the same shape as each other, likewise `b`/`grad_b`), so one function
/// covers both the single-example (`batch_size=1`) and batched caller.
#[pyfunction]
pub fn layer_apply_accumulated_gradient(
    w: &RustArray,
    b: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    learning_rate: f64,
    batch_size: usize,
) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::fused::layer_apply_accumulated_gradient(w, b, grad_w, grad_b, learning_rate, batch_size)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

/// One single-example plain-SGD step: `layer_accumulate_gradient` into fresh zero accumulators,
/// then `layer_apply_accumulated_gradient` with `batch_size=1`, in one pass and one allocation
/// per parameter. `RustArrayNetworkBase.learn` runs exactly that pair on every layer, and apply
/// resets the accumulators, so at batch size 1 they are always fresh. Computes `w - lr * (0.0 +
/// delta[i] * x[j])`: the `0.0 +` is the add into the zero accumulator, kept on purpose. It turns
/// a `-0.0` product into `+0.0`, which changes the result where `w` is `-0.0`, so dropping it
/// would break bit-identity with the pair it replaces.
#[pyfunction]
pub fn layer_sgd_step(
    w: &RustArray,
    b: &RustArray,
    delta: &RustArray,
    input_activation: &RustArray,
    learning_rate: f64,
) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::fused::layer_sgd_step(w, b, delta, input_activation, learning_rate)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (w, b, grad_w, grad_b, m_w, v_w, m_b, v_b, t, beta1, beta2, epsilon, learning_rate, batch_size, weight_decay=0.0))]
pub fn layer_adam_apply_accumulated_gradient(
    w: &RustArray,
    b: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    m_w: &RustArray,
    v_w: &RustArray,
    m_b: &RustArray,
    v_b: &RustArray,
    t: u32,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    learning_rate: f64,
    batch_size: usize,
    weight_decay: f64,
) -> PyResult<(RustArray, RustArray, RustArray, RustArray, RustArray, RustArray)> {
    indrajala_math::fused::layer_adam_apply_accumulated_gradient(
        w,
        b,
        grad_w,
        grad_b,
        m_w,
        v_w,
        m_b,
        v_b,
        t,
        beta1,
        beta2,
        epsilon,
        learning_rate,
        batch_size,
        weight_decay,
    )
    .map(|r| {
        let (r0, r1, r2, r3, r4, r5) = r;
        (
            RustArray::from(r0),
            RustArray::from(r1),
            RustArray::from(r2),
            RustArray::from(r3),
            RustArray::from(r4),
            RustArray::from(r5),
        )
    })
    .map_err(py_err)
}

/// `L2ArrayLayer.apply_accumulated_gradient`: weight decay as Goyal et al. 2017's eq. (8), `λW`
/// added to the mean gradient: `W -= learning_rate * (grad_W / batch_size + l2_lambda * W);
/// b -= learning_rate * (grad_b / batch_size)` (bias unregularized). No persistent
/// per-parameter state at all (unlike
/// `layer_adam_apply_accumulated_gradient`/`layer_momentum_apply_accumulated_gradient`), so this
/// takes only `W`/`b`/`grad_W`/`grad_b` plus the scalar `l2_lambda` - the simplest fused op in
/// this round.
#[pyfunction]
pub fn layer_l2_apply_accumulated_gradient(
    w: &RustArray,
    b: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    l2_lambda: f64,
    learning_rate: f64,
    batch_size: usize,
) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::fused::layer_l2_apply_accumulated_gradient(
        w,
        b,
        grad_w,
        grad_b,
        l2_lambda,
        learning_rate,
        batch_size,
    )
    .map(|r| {
        let (r0, r1) = r;
        (RustArray::from(r0), RustArray::from(r1))
    })
    .map_err(py_err)
}

/// `MomentumArrayLayer.apply_accumulated_gradient`: momentum as Goyal et al. 2017's eq. (9),
/// `velocity = momentum * velocity + grad / batch_size; param -= learning_rate * velocity`. It
/// replaces Rumelhart et al.'s eq. (10), which folds the rate into the velocity and so needs a
/// correction when the rate changes. One velocity array per parameter tensor, shape-agnostic like
/// `layer_apply_accumulated_gradient` (covers both the `W`/`grad_W`/`velocity_W` (2D) and
/// `b`/`grad_b`/`velocity_b` (1D) cases via two calls from the Python caller, the same convention
/// `layer_apply_accumulated_gradient` itself uses).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn layer_momentum_apply_accumulated_gradient(
    w: &RustArray,
    b: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    velocity_w: &RustArray,
    velocity_b: &RustArray,
    momentum: f64,
    learning_rate: f64,
    batch_size: usize,
) -> PyResult<(RustArray, RustArray, RustArray, RustArray)> {
    indrajala_math::fused::layer_momentum_apply_accumulated_gradient(
        w,
        b,
        grad_w,
        grad_b,
        velocity_w,
        velocity_b,
        momentum,
        learning_rate,
        batch_size,
    )
    .map(|r| {
        let (r0, r1, r2, r3) = r;
        (
            RustArray::from(r0),
            RustArray::from(r1),
            RustArray::from(r2),
            RustArray::from(r3),
        )
    })
    .map_err(py_err)
}

/// `ReLUArrayLayer.forward`: `max(0, self.W @ x + self.b)`, `x`/`b` both 1D. Inlines
/// `array_relu`'s own formula (`ufuncs.rs`) rather than calling it as a separate op, the same
/// "one Rust call per layer method" discipline every other `fused.rs` function follows.
#[pyfunction]
pub fn layer_relu_forward(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::fused::layer_relu_forward(w, x, b)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ReLUArrayLayer.forward_batch`: `max(0, X @ self.W.T + self.b)`, `X` 2D (`batch, input_size`).
#[pyfunction]
pub fn layer_relu_forward_batch(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::fused::layer_relu_forward_batch(w, x, b)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ReLUArrayLayer.compute_hidden_delta`: `(next_layer.W.T @ next_layer.delta) * (self.a > 0.0)` -
/// inlines `array_relu_mask`'s own formula rather than calling it as a separate op, single-example
/// (`next_delta`/`a` both 1D).
#[pyfunction]
pub fn layer_relu_hidden_delta(next_w: &RustArray, next_delta: &RustArray, a: &RustArray) -> PyResult<RustArray> {
    indrajala_math::fused::layer_relu_hidden_delta(next_w, next_delta, a)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ReLUArrayLayer.compute_hidden_delta_batch`: `(next_layer.delta_batch @ next_layer.W) *
/// (self.A > 0.0)` - batched, no transpose on `next_w` (same shape-of-call-sites distinction
/// `layer_hidden_delta_batch` itself already documents).
#[pyfunction]
pub fn layer_relu_hidden_delta_batch(
    next_w: &RustArray,
    next_delta_batch: &RustArray,
    a_batch: &RustArray,
) -> PyResult<RustArray> {
    indrajala_math::fused::layer_relu_hidden_delta_batch(next_w, next_delta_batch, a_batch)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `SoftmaxArrayLayer.forward`: `array_softmax(self.W @ x + self.b)`, `x`/`b` both 1D. Reuses
/// `array_softmax` (`ufuncs.rs`) directly rather than inlining its max/sum reduction, unlike
/// ReLU's trivial elementwise formula - the extra call is Rust-internal, not a second
/// Python/Rust FFI crossing.
#[pyfunction]
pub fn layer_softmax_forward(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::fused::layer_softmax_forward(w, x, b)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `SoftmaxArrayLayer.forward_batch`: `array_softmax(X @ self.W.T + self.b)`, row-wise
/// normalization, `X` 2D (`batch, input_size`).
#[pyfunction]
pub fn layer_softmax_forward_batch(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::fused::layer_softmax_forward_batch(w, x, b)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `SoftmaxArrayLayer.compute_output_delta`/`compute_output_delta_batch`: `a - reference` -
/// `SoftmaxOutputNode`'s own simplification, no `a*(1-a)` damping term at all - shape-agnostic
/// like `layer_output_delta`, so one function covers both the single-example and batched case.
#[pyfunction]
pub fn layer_softmax_output_delta(a: &RustArray, reference: &RustArray) -> PyResult<RustArray> {
    indrajala_math::fused::layer_softmax_output_delta(a, reference)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `DropoutArrayLayer.forward`: `sigmoid(self.W @ x + self.b)`, with a training-time
/// inverted-dropout mask drawn internally, from `rng` if given, else from `random.rs`'s global
/// MT19937 (`draw_bernoulli_mask`). Returns
/// `(a, mask, base_activation)`: `DropoutRustArrayLayer`
/// keeps `mask`/`base_activation` around as this layer's own forward-time snapshots for
/// `layer_dropout_hidden_delta` below, the same role `_mask`/`_base_activation` play on the
/// numpy-backed `DropoutArrayLayer`. `x`/`b` both 1D; at `training=false` the mask is all-ones
/// and `a == base_activation` exactly, no rescale - matching `DropoutNode.forward`'s own
/// eval-mode no-op.
#[pyfunction]
#[pyo3(signature = (w, x, b, drop_probability, training, rng=None))]
pub fn layer_dropout_forward(
    w: &RustArray,
    x: &RustArray,
    b: &RustArray,
    drop_probability: f64,
    training: bool,
    mut rng: Option<PyRefMut<'_, Generator>>,
) -> PyResult<(RustArray, RustArray, RustArray)> {
    indrajala_math::fused::layer_dropout_forward(
        w,
        x,
        b,
        drop_probability,
        training,
        rng.as_deref_mut().map(|g| &mut g.0),
    )
    .map(|r| {
        let (r0, r1, r2) = r;
        (RustArray::from(r0), RustArray::from(r1), RustArray::from(r2))
    })
    .map_err(py_err)
}

/// `DropoutArrayLayer.forward_batch`: `sigmoid(X @ self.W.T + self.b)`, with one independent
/// mask entry per (example, unit) pair (`draw_bernoulli_mask`'s flat draw over the whole
/// `batch_size * size` buffer) - matching `forward`'s own per-example-independent-draw
/// requirement, not one mask shared across the batch. `X` 2D (`batch, input_size`).
#[pyfunction]
#[pyo3(signature = (w, x, b, drop_probability, training, rng=None))]
pub fn layer_dropout_forward_batch(
    w: &RustArray,
    x: &RustArray,
    b: &RustArray,
    drop_probability: f64,
    training: bool,
    mut rng: Option<PyRefMut<'_, Generator>>,
) -> PyResult<(RustArray, RustArray, RustArray)> {
    indrajala_math::fused::layer_dropout_forward_batch(
        w,
        x,
        b,
        drop_probability,
        training,
        rng.as_deref_mut().map(|g| &mut g.0),
    )
    .map(|r| {
        let (r0, r1, r2) = r;
        (RustArray::from(r0), RustArray::from(r1), RustArray::from(r2))
    })
    .map_err(py_err)
}

/// `DropoutArrayLayer.compute_hidden_delta`: `(next_layer.W.T @ next_layer.delta) *
/// base_activation*(1-base_activation) * scale`, where `scale = mask/keep_probability` if
/// `was_training` else `1.0`. `base_activation`/`mask` are the forward-time snapshots
/// `layer_dropout_forward` returned, not re-derived here; `was_training` is a forward-time
/// snapshot of `training` too, not a live re-read - mirrors `DropoutNode.compute_hidden_delta`'s
/// own `_was_training` subtlety: the caller's own `training` flag is already back to `false` by
/// the time backward runs, so the rescale must be decided from what `forward` actually did.
/// Single-example (`next_delta`/`base_activation`/`mask` all 1D).
#[pyfunction]
pub fn layer_dropout_hidden_delta(
    next_w: &RustArray,
    next_delta: &RustArray,
    base_activation: &RustArray,
    mask: &RustArray,
    keep_probability: f64,
    was_training: bool,
) -> PyResult<RustArray> {
    indrajala_math::fused::layer_dropout_hidden_delta(
        next_w,
        next_delta,
        base_activation,
        mask,
        keep_probability,
        was_training,
    )
    .map(RustArray::from)
    .map_err(py_err)
}

/// `DropoutArrayLayer.compute_hidden_delta_batch`: batched, no transpose on `next_w` (same
/// shape-of-call-sites distinction `layer_hidden_delta_batch`/`layer_relu_hidden_delta_batch`
/// already document).
#[pyfunction]
pub fn layer_dropout_hidden_delta_batch(
    next_w: &RustArray,
    next_delta_batch: &RustArray,
    base_activation_batch: &RustArray,
    mask_batch: &RustArray,
    keep_probability: f64,
    was_training: bool,
) -> PyResult<RustArray> {
    indrajala_math::fused::layer_dropout_hidden_delta_batch(
        next_w,
        next_delta_batch,
        base_activation_batch,
        mask_batch,
        keep_probability,
        was_training,
    )
    .map(RustArray::from)
    .map_err(py_err)
}

/// `AffineArrayLayer.forward`: `self.W @ x + self.b` with no activation, the affine layer that ends
/// a residual block's body (indrajala-ml's residual-connections workplan, D4): `linear_preactivation`
/// itself, so its bits are the pre-activation every `layer_*forward` computes.
#[pyfunction]
pub fn affine_forward(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::fused::affine_forward(w, x, b)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `AffineArrayLayer.forward_batch`: `X @ self.W.T + self.b`, `linear_preactivation_batch`, so row
/// `i` is bit-identical to `affine_forward(w, X[i], b)`.
#[pyfunction]
pub fn affine_forward_batch(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::fused::affine_forward_batch(w, x, b)
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
    indrajala_math::fused::layer_hidden_delta_skip(body_w, body_delta, skip, a)
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
    indrajala_math::fused::layer_hidden_delta_skip_batch(body_w, body_delta_batch, skip_batch, a_batch)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ReLUArrayLayer.compute_hidden_delta` when the next layer is a residual block's fork: `(body_w.T
/// @ body_delta + skip) * (self.a > 0.0)`, `layer_relu_hidden_delta` with the skip term.
#[pyfunction]
pub fn layer_relu_hidden_delta_skip(
    body_w: &RustArray,
    body_delta: &RustArray,
    skip: &RustArray,
    a: &RustArray,
) -> PyResult<RustArray> {
    indrajala_math::fused::layer_relu_hidden_delta_skip(body_w, body_delta, skip, a)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `layer_relu_hidden_delta_skip`, batched: `(body_delta_batch @ body_w + skip_batch) * (self.A >
/// 0.0)`.
#[pyfunction]
pub fn layer_relu_hidden_delta_skip_batch(
    body_w: &RustArray,
    body_delta_batch: &RustArray,
    skip_batch: &RustArray,
    a_batch: &RustArray,
) -> PyResult<RustArray> {
    indrajala_math::fused::layer_relu_hidden_delta_skip_batch(body_w, body_delta_batch, skip_batch, a_batch)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `DropoutArrayLayer.compute_hidden_delta` when the next layer is a residual block's fork:
/// `layer_dropout_hidden_delta` with the skip term added to the downstream before the sigmoid's
/// derivative and the mask's scale.
#[pyfunction]
pub fn layer_dropout_hidden_delta_skip(
    body_w: &RustArray,
    body_delta: &RustArray,
    skip: &RustArray,
    base_activation: &RustArray,
    mask: &RustArray,
    keep_probability: f64,
    was_training: bool,
) -> PyResult<RustArray> {
    indrajala_math::fused::layer_dropout_hidden_delta_skip(
        body_w,
        body_delta,
        skip,
        base_activation,
        mask,
        keep_probability,
        was_training,
    )
    .map(RustArray::from)
    .map_err(py_err)
}

/// `layer_dropout_hidden_delta_skip`, batched.
#[pyfunction]
pub fn layer_dropout_hidden_delta_skip_batch(
    body_w: &RustArray,
    body_delta_batch: &RustArray,
    skip_batch: &RustArray,
    base_activation_batch: &RustArray,
    mask_batch: &RustArray,
    keep_probability: f64,
    was_training: bool,
) -> PyResult<RustArray> {
    indrajala_math::fused::layer_dropout_hidden_delta_skip_batch(
        body_w,
        body_delta_batch,
        skip_batch,
        base_activation_batch,
        mask_batch,
        keep_probability,
        was_training,
    )
    .map(RustArray::from)
    .map_err(py_err)
}
