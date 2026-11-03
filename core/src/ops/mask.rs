//! The masks the backward passes apply: `downstream` times an activation's derivative, sigmoid's,
//! or dropout's mask over sigmoid's.

use crate::array::Array;
use crate::num::activation::Activation;
use crate::ops::arith::{require_same_shape, same_shape_elementwise};
use crate::Error;

/// `downstream * act'(cached)`, shape-checked, for the fused deltas; [`activation`] is its public form.
/// `cached` is the layer's `a` for ReLU, its `z` for the GELUs.
pub(crate) fn activation_mask(
    downstream: &Array,
    cached: &Array,
    activation: Activation,
    context: &str,
) -> Result<Array, Error> {
    require_same_shape(downstream, cached, context)?;
    Ok(masked(downstream, cached, activation))
}

/// `downstream * act'(cached)` for a downstream whose shape `hidden_downstream*` (and `with_skip`)
/// already checked against `cached`: the hidden deltas' last step, no second check.
pub(crate) fn masked(downstream: &Array, cached: &Array, activation: Activation) -> Array {
    Array {
        data: activation.mask(&downstream.data, &cached.data),
        shape: cached.shape,
    }
}

/// Shared by `nn::dropout::hidden_delta`/`hidden_delta_batch` and by [`dropout`] - they differ only
/// in how `downstream` was computed, not in the elementwise formula applied on top of it.
pub(crate) fn dropout_hidden_delta_from_downstream(
    downstream: &Array,
    base_activation: &Array,
    mask: &Array,
    keep_probability: f64,
    was_training: bool,
) -> Array {
    let data = downstream
        .data
        .iter()
        .zip(base_activation.data.iter())
        .zip(mask.data.iter())
        .map(|((&d, &base_value), &mask_value)| {
            let sigmoid_derivative = base_value * (1.0 - base_value);
            let scale = if was_training {
                mask_value / keep_probability
            } else {
                1.0
            };
            d * sigmoid_derivative * scale
        })
        .collect();
    Array {
        data,
        shape: base_activation.shape,
    }
}

/// `downstream * act'(cached)` - a hidden layer's delta from a downstream it was handed (before a
/// layer with no `W` to read: a `LayerNorm`, a token-wise layer, a conv layer's next), the fused
/// `layer_activation_hidden_delta*` ops' own elementwise step, so given the same downstream it
/// computes their bits. `cached` is the layer's `a` for ReLU (the mask `a > 0`), its `z` for the
/// GELUs ([`Activation::keeps_preactivation`]). Shape-agnostic (single-example 1D or batched 2D).
pub fn activation(downstream: &Array, cached: &Array, activation: Activation) -> Result<Array, Error> {
    activation_mask(downstream, cached, activation, "array_activation_mask")
}

/// `downstream * a * (1 - a)` - the sigmoid's derivative applied to a downstream that a dense
/// layer's fused `layer_hidden_delta*` can't compute, because the next layer has no `W` to read
/// (a `LayerNorm`, or a residual fork whose body starts with one: indrajala-ml's layer-norm and
/// attention workplan, D5). The fused ops' elementwise expression, in their grouping, so given the
/// same downstream it computes their bits. Shape-agnostic, as `activation`.
pub fn sigmoid(downstream: &Array, a: &Array) -> Result<Array, Error> {
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
/// downstream it was handed, as `sigmoid` is the sigmoid layer's (D5): the fused
/// `layer_dropout_hidden_delta*` ops' own elementwise expression. Shape-agnostic.
pub fn dropout(
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
