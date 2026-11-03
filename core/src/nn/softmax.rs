//! The softmax output layer (`SoftmaxArrayLayer`): its forward and its output delta.

use crate::array::Array;
use crate::nn::dense::{linear_preactivation, linear_preactivation_batch};
use crate::ops::arith::{require_same_shape, same_shape_elementwise};
use crate::ops::unary::array_softmax;
use crate::Error;

/// `SoftmaxArrayLayer.forward`: `array_softmax(self.W @ x + self.b)`, `x`/`b` both 1D. Reuses
/// `array_softmax` (`ufuncs.rs`) directly rather than inlining its max/sum reduction, unlike
/// ReLU's trivial elementwise formula - the extra call is Rust-internal, not a second
/// Python/Rust FFI crossing.
pub fn layer_softmax_forward(w: &Array, x: &Array, b: &Array) -> Result<Array, Error> {
    Ok(array_softmax(&linear_preactivation(w, x, b)?))
}

/// `SoftmaxArrayLayer.forward_batch`: `array_softmax(X @ self.W.T + self.b)`, row-wise
/// normalization, `X` 2D (`batch, input_size`).
pub fn layer_softmax_forward_batch(w: &Array, x: &Array, b: &Array) -> Result<Array, Error> {
    Ok(array_softmax(&linear_preactivation_batch(w, x, b)?))
}

/// `SoftmaxArrayLayer.compute_output_delta`/`compute_output_delta_batch`: `a - reference` -
/// `SoftmaxOutputNode`'s own simplification, no `a*(1-a)` damping term at all - shape-agnostic
/// like `layer_output_delta`, so one function covers both the single-example and batched case.
pub fn layer_softmax_output_delta(a: &Array, reference: &Array) -> Result<Array, Error> {
    require_same_shape(a, reference, "layer_softmax_output_delta")?;
    Ok(Array {
        data: same_shape_elementwise(&a.data, &reference.data, |av, rv| av - rv),
        shape: a.shape,
    })
}
