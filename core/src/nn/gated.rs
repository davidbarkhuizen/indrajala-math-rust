//! The gated token-wise layer (indrajala-ml's FFN activations workplan, D3): `act(gate) * value`,
//! its forward and its delta.

use crate::array::{Array, Shape};
use crate::num::activation::Activation;
use crate::nn::dense::linear_preactivation_batch;
use crate::Error;

/// A gated token-wise layer's split of its pre-activation `z` `(rows, 2 * size)` into the gate
/// `a` (each row's first `size` columns) and the value `v` (its last `size`), row-major, and `size`
/// (indrajala-ml's FFN activations workplan, D3).
fn gate_and_value(z: &Array, context: &str) -> Result<(Vec<f64>, Vec<f64>, usize, usize), Error> {
    let Shape::Matrix(rows, width) = z.shape else {
        return Err(Error::invalid(format!(
            "{context} requires a 2D z (rows, 2 * size), got {:?}",
            z.shape
        )));
    };
    if width == 0 || width % 2 != 0 {
        return Err(Error::invalid(format!(
            "{context} requires an even, non-zero width (the gate's and the value's), got {width}"
        )));
    }
    let size = width / 2;
    let (mut gate, mut value) = (Vec::with_capacity(rows * size), Vec::with_capacity(rows * size));
    for row in z.data.chunks_exact(width) {
        gate.extend_from_slice(&row[..size]);
        value.extend_from_slice(&row[size..]);
    }
    Ok((gate, value, rows, size))
}

/// `GatedDenseArrayLayer.forward_batch` on its rows (indrajala-ml's FFN activations workplan, D3;
/// Shazeer 2020's `FFN_GLU` family): `z = X @ W.T + b`, `W` `(2 * size, input_size)` the gate's
/// rows then the value's, and `act(a) * v` of each row's gate `a` and value `v`, `act` SiLU for
/// SwiGLU, or a GELU for GEGLU. One product for both projections. Returns `(out, z)`: `out`
/// `(rows, size)`, and `z` `(rows, 2 * size)`, which the backward reads.
pub fn gated_forward_batch(w: &Array, x: &Array, b: &Array, activation: Activation) -> Result<(Array, Array), Error> {
    let z = linear_preactivation_batch(w, x, b)?;
    let (gate, value, rows, size) = gate_and_value(&z, "gated_forward_batch")?;
    let data = activation
        .apply(&gate)
        .iter()
        .zip(value.iter())
        .map(|(&g, &v)| g * v)
        .collect();
    Ok((Array::from_matrix(data, rows, size), z))
}

/// `GatedDenseArrayLayer`'s delta of its pre-activation from its output's, `delta` `(rows, size)`:
/// each row's gate part `(d * v) * act'(a)` then its value part `d * act(a)`, `(rows, 2 * size)`,
/// laid out as `z`. The downstream and the gradients are then the dense ops'
/// (`layer_downstream_batch`, `layer_accumulate_gradient_batch`) on it.
pub fn gated_delta_batch(delta: &Array, z: &Array, activation: Activation) -> Result<Array, Error> {
    let context = "gated_delta_batch";
    let (gate, value, rows, size) = gate_and_value(z, context)?;
    if delta.shape != Shape::Matrix(rows, size) {
        return Err(Error::invalid(format!(
            "{context} requires a delta of shape {:?} (z's rows, half its width), got {:?}",
            Shape::Matrix(rows, size),
            delta.shape
        )));
    }
    let scaled: Vec<f64> = delta.data.iter().zip(value.iter()).map(|(&d, &v)| d * v).collect();
    let gate_part = activation.mask(&scaled, &gate);
    let activated = activation.apply(&gate);
    let mut data = Vec::with_capacity(rows * 2 * size);
    for r in 0..rows {
        let span = r * size..(r + 1) * size;
        data.extend_from_slice(&gate_part[span.clone()]);
        data.extend(
            delta.data[span.clone()]
                .iter()
                .zip(activated[span].iter())
                .map(|(&d, &g)| d * g),
        );
    }
    Ok(Array::from_matrix(data, rows, 2 * size))
}
