//! The bindings of the core's `nn::dropout`: the dropout dense layer.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::error::py_err;
use crate::random::generator::Generator;

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
    indrajala_math::nn::dropout::forward(
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
    indrajala_math::nn::dropout::forward_batch(
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
    indrajala_math::nn::dropout::hidden_delta(
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
/// shape-of-call-sites distinction `layer_hidden_delta_batch`/`layer_activation_hidden_delta_batch`
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
    indrajala_math::nn::dropout::hidden_delta_batch(
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
    indrajala_math::nn::dropout::hidden_delta_skip(
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
    indrajala_math::nn::dropout::hidden_delta_skip_batch(
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

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(layer_dropout_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layer_dropout_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_dropout_hidden_delta, m)?)?;
    m.add_function(wrap_pyfunction!(layer_dropout_hidden_delta_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_dropout_hidden_delta_skip, m)?)?;
    m.add_function(wrap_pyfunction!(layer_dropout_hidden_delta_skip_batch, m)?)?;
    Ok(())
}
