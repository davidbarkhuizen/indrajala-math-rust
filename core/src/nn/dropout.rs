//! The dropout dense layer (`DropoutArrayLayer`): a sigmoid layer with an inverted-dropout mask
//! drawn in training, its hidden deltas and their skip variants.

use crate::array::Array;
use crate::nn::dense::{
    hidden_downstream, hidden_downstream_batch, linear_preactivation, linear_preactivation_batch, sigmoid, with_skip,
};
use crate::ops::mask::dropout_hidden_delta_from_downstream;
use crate::random::generator::Generator;
use crate::random::legacy::draw_bernoulli_mask;
use crate::Error;

/// `DropoutArrayLayer.forward`: `sigmoid(self.W @ x + self.b)`, with a training-time
/// inverted-dropout mask drawn internally, from `rng` if given, else from `random.rs`'s global
/// MT19937 (`draw_bernoulli_mask`). Returns
/// `(a, mask, base_activation)`: `DropoutRustArrayLayer`
/// keeps `mask`/`base_activation` around as this layer's own forward-time snapshots for
/// `hidden_delta` below, the same role `_mask`/`_base_activation` play on the
/// numpy-backed `DropoutArrayLayer`. `x`/`b` both 1D; at `training=false` the mask is all-ones
/// and `a == base_activation` exactly, no rescale - matching `DropoutNode.forward`'s own
/// eval-mode no-op.
pub fn forward(
    w: &Array,
    x: &Array,
    b: &Array,
    drop_probability: f64,
    training: bool,
    rng: Option<&mut Generator>,
) -> Result<(Array, Array, Array), Error> {
    let z = linear_preactivation(w, x, b)?;
    let base = sigmoid(&z);
    let (a, mask) = dropout_forward_from_base(&base, drop_probability, training, rng);
    Ok((a, mask, base))
}

/// `DropoutArrayLayer.forward_batch`: `sigmoid(X @ self.W.T + self.b)`, with one independent
/// mask entry per (example, unit) pair (`draw_bernoulli_mask`'s flat draw over the whole
/// `batch_size * size` buffer) - matching `forward`'s own per-example-independent-draw
/// requirement, not one mask shared across the batch. `X` 2D (`batch, input_size`).
pub fn forward_batch(
    w: &Array,
    x: &Array,
    b: &Array,
    drop_probability: f64,
    training: bool,
    rng: Option<&mut Generator>,
) -> Result<(Array, Array, Array), Error> {
    let z = linear_preactivation_batch(w, x, b)?;
    let base = sigmoid(&z);
    let (a, mask) = dropout_forward_from_base(&base, drop_probability, training, rng);
    Ok((a, mask, base))
}

/// Shared by `forward`/`forward_batch` above - both differ only in
/// how `base` (the pre-mask sigmoid) was computed (single-example matvec vs. batched matmul),
/// not in how the mask is drawn and applied on top of it. The mask comes from `rng` (a PCG64
/// `Generator`) when one is passed, else from `random.rs`'s global MT19937.
fn dropout_forward_from_base(
    base: &Array,
    drop_probability: f64,
    training: bool,
    rng: Option<&mut Generator>,
) -> (Array, Array) {
    let keep_probability = 1.0 - drop_probability;
    let size = base.data.len();
    if training {
        let mask_data = match rng {
            Some(rng) => rng.draw_bernoulli_mask(drop_probability, size),
            None => draw_bernoulli_mask(drop_probability, size),
        };
        let a_data: Vec<f64> = base
            .data
            .iter()
            .zip(mask_data.iter())
            .map(|(&bv, &mv)| bv * mv / keep_probability)
            .collect();
        (
            Array {
                data: a_data,
                shape: base.shape,
            },
            Array {
                data: mask_data,
                shape: base.shape,
            },
        )
    } else {
        (
            base.clone(),
            Array {
                data: vec![1.0; size],
                shape: base.shape,
            },
        )
    }
}

/// `DropoutArrayLayer.compute_hidden_delta`: `(next_layer.W.T @ next_layer.delta) *
/// base_activation*(1-base_activation) * scale`, where `scale = mask/keep_probability` if
/// `was_training` else `1.0`. `base_activation`/`mask` are the forward-time snapshots
/// `forward` returned, not re-derived here; `was_training` is a forward-time
/// snapshot of `training` too, not a live re-read - mirrors `DropoutNode.compute_hidden_delta`'s
/// own `_was_training` subtlety: the caller's own `training` flag is already back to `false` by
/// the time backward runs, so the rescale must be decided from what `forward` actually did.
/// Single-example (`next_delta`/`base_activation`/`mask` all 1D).
pub fn hidden_delta(
    next_w: &Array,
    next_delta: &Array,
    base_activation: &Array,
    mask: &Array,
    keep_probability: f64,
    was_training: bool,
) -> Result<Array, Error> {
    let downstream = hidden_downstream(next_w, next_delta, base_activation, "layer_dropout_hidden_delta")?;
    Ok(dropout_hidden_delta_from_downstream(
        &downstream,
        base_activation,
        mask,
        keep_probability,
        was_training,
    ))
}

/// `DropoutArrayLayer.compute_hidden_delta_batch`: batched, no transpose on `next_w` (same
/// shape-of-call-sites distinction `dense::hidden_delta_batch`/`dense::activation_hidden_delta_batch`
/// already document).
pub fn hidden_delta_batch(
    next_w: &Array,
    next_delta_batch: &Array,
    base_activation_batch: &Array,
    mask_batch: &Array,
    keep_probability: f64,
    was_training: bool,
) -> Result<Array, Error> {
    let downstream = hidden_downstream_batch(
        next_w,
        next_delta_batch,
        base_activation_batch,
        "layer_dropout_hidden_delta_batch",
    )?;
    Ok(dropout_hidden_delta_from_downstream(
        &downstream,
        base_activation_batch,
        mask_batch,
        keep_probability,
        was_training,
    ))
}

/// `DropoutArrayLayer.compute_hidden_delta` when the next layer is a residual block's fork:
/// `hidden_delta` with the skip term added to the downstream before the sigmoid's
/// derivative and the mask's scale.
pub fn hidden_delta_skip(
    body_w: &Array,
    body_delta: &Array,
    skip: &Array,
    base_activation: &Array,
    mask: &Array,
    keep_probability: f64,
    was_training: bool,
) -> Result<Array, Error> {
    let downstream = hidden_downstream(body_w, body_delta, base_activation, "layer_dropout_hidden_delta_skip")?;
    let downstream = with_skip(downstream, skip, "layer_dropout_hidden_delta_skip")?;
    Ok(dropout_hidden_delta_from_downstream(
        &downstream,
        base_activation,
        mask,
        keep_probability,
        was_training,
    ))
}

/// `hidden_delta_skip`, batched.
pub fn hidden_delta_skip_batch(
    body_w: &Array,
    body_delta_batch: &Array,
    skip_batch: &Array,
    base_activation_batch: &Array,
    mask_batch: &Array,
    keep_probability: f64,
    was_training: bool,
) -> Result<Array, Error> {
    let downstream = hidden_downstream_batch(
        body_w,
        body_delta_batch,
        base_activation_batch,
        "layer_dropout_hidden_delta_skip_batch",
    )?;
    let downstream = with_skip(downstream, skip_batch, "layer_dropout_hidden_delta_skip_batch")?;
    Ok(dropout_hidden_delta_from_downstream(
        &downstream,
        base_activation_batch,
        mask_batch,
        keep_probability,
        was_training,
    ))
}
