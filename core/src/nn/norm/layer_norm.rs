//! Layer norm (indrajala-ml's layer-norm and attention workplan, D5): one Rust function per method
//! of `LayerNormArrayLayer` in `indrajala_ml/model/layers/numpy/layer_norm_array_layer.py`, as [`batch_norm`](super::batch_norm)
//! is for batch norm. Each token's `d` features (a flat layer's whole vector being one token) are
//! normalized by their own mean and biased variance, then `gamma * xhat + beta`, `gamma` and
//! `beta` of length `d` shared over the tokens. A token-major batch of `T` tokens per example is
//! `N * T` rows of `d` without a copy, so the functions only ever walk rows, in (example, token)
//! order.
//!
//! The expressions are indrajala-ml's README's (Layer norm and attention), in their grouping, left
//! to right, with only IEEE 754's correctly rounded `+ - * /` and `sqrt`: no `mul_add` and no
//! `powf`. Every sum is a left fold from `0.0`, over a row's features for the statistics and the
//! backward pass's means, over the rows for the gradients, as numpy's `np.cumsum` along the axis.
//! So given the same inputs they compute numpy's bits.

use crate::array::{Array, Shape};
use crate::Error;

/// A left fold from `0.0` in index order.
fn fold(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, |acc, v| acc + v)
}

/// The number of `features`-wide rows in `x`, checked to be whole.
fn feature_rows(x: &Array, features: usize, name: &str, context: &str) -> Result<usize, Error> {
    let n = x.data.len();
    if features == 0 || !n.is_multiple_of(features) {
        return Err(Error::invalid(format!(
            "{context} requires {name} of whole tokens of {features} features (gamma's length), got shape {:?}",
            x.shape
        )));
    }
    Ok(n / features)
}

fn require_shape(array: &Array, shape: Shape, name: &str, context: &str) -> Result<(), Error> {
    if array.shape != shape {
        return Err(Error::invalid(format!(
            "{context} requires {name} of shape {shape:?}, got {:?}",
            array.shape
        )));
    }
    Ok(())
}

/// `normalize`'s rows. Only the two folds are chains; the rest is one correctly rounded operation per
/// value, the same bits in any lane width, with `xhat`'s row holding the centred values until `s` is known.
#[inline(always)]
fn normalize_rows(
    x: &[f64],
    gamma: &[f64],
    beta: &[f64],
    epsilon: f64,
    a: &mut [f64],
    xhat: &mut [f64],
    std: &mut [f64],
) {
    let d = gamma.len();
    let rows = x
        .chunks_exact(d)
        .zip(a.chunks_exact_mut(d))
        .zip(xhat.chunks_exact_mut(d));
    for (((row, a_row), xhat_row), std_value) in rows.zip(std) {
        let mu = fold(row.iter().copied()) / d as f64;
        for (c, &v) in xhat_row.iter_mut().zip(row) {
            *c = v - mu;
        }
        let var = fold(xhat_row.iter().map(|&v| v * v)) / d as f64;
        let s = (var + epsilon).sqrt();
        for xh in xhat_row.iter_mut() {
            *xh /= s;
        }
        for ((a_value, &xh), (&g, &b)) in a_row.iter_mut().zip(&*xhat_row).zip(gamma.iter().zip(beta)) {
            *a_value = g * xh + b;
        }
        *std_value = s;
    }
}

/// `normalize_rows` with four lanes. AVX2 alone: no FMA, whose fused `g * xh + b` would round once.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_rows_avx2(
    x: &[f64],
    gamma: &[f64],
    beta: &[f64],
    epsilon: f64,
    a: &mut [f64],
    xhat: &mut [f64],
    std: &mut [f64],
) {
    normalize_rows(x, gamma, beta, epsilon, a, xhat, std);
}

/// The forward pass over `x`'s rows: `(a, xhat, std)`, `a` and `xhat` in `x`'s shape and `std`
/// one value per row.
fn normalize(
    x: &Array,
    gamma: &Array,
    beta: &Array,
    epsilon: f64,
    context: &str,
) -> Result<(Array, Array, Array), Error> {
    let d = gamma.data.len();
    let rows = feature_rows(x, d, "x", context)?;
    require_shape(beta, Shape::Vector(d), "beta", context)?;
    let mut a = vec![0.0; x.data.len()];
    let mut xhat = vec![0.0; x.data.len()];
    let mut std = vec![0.0; rows];
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { normalize_rows_avx2(&x.data, &gamma.data, &beta.data, epsilon, &mut a, &mut xhat, &mut std) };
    } else {
        normalize_rows(&x.data, &gamma.data, &beta.data, epsilon, &mut a, &mut xhat, &mut std);
    }
    #[cfg(not(target_arch = "x86_64"))]
    normalize_rows(&x.data, &gamma.data, &beta.data, epsilon, &mut a, &mut xhat, &mut std);
    Ok((
        Array {
            data: a,
            shape: x.shape,
        },
        Array {
            data: xhat,
            shape: x.shape,
        },
        Array::from_vector(std),
    ))
}

/// `LayerNormArrayLayer.forward`: one example, `x` 1D of `T * d` values. Returns `(a, xhat, std)`,
/// what the backward pass reads: `xhat` in `x`'s layout and `std = sqrt(var + eps)` per token.
/// Training and inference alike.
pub fn forward(x: &Array, gamma: &Array, beta: &Array, epsilon: f64) -> Result<(Array, Array, Array), Error> {
    let context = "layer_norm_forward";
    if !matches!(x.shape, Shape::Vector(_)) {
        return Err(Error::invalid(format!(
            "{context} requires a 1D x, got shape {:?}",
            x.shape
        )));
    }
    normalize(x, gamma, beta, epsilon, context)
}

/// `LayerNormArrayLayer.forward_batch`: `x` 2D (`batch, T * d`), each row `forward`'s
/// bits. Returns `(a, xhat, std)`, `std` per (example, token), 1D of `batch * T`.
pub fn forward_batch(x: &Array, gamma: &Array, beta: &Array, epsilon: f64) -> Result<(Array, Array, Array), Error> {
    let context = "layer_norm_forward_batch";
    if !matches!(x.shape, Shape::Matrix(_, _)) {
        return Err(Error::invalid(format!(
            "{context} requires a 2D x, got shape {:?}",
            x.shape
        )));
    }
    normalize(x, gamma, beta, epsilon, context)
}

/// `LayerNormArrayLayer.downstream_batch`: `dl/dx` from `delta_batch` (`dl/dy`), per row,
/// `dxhat = delta * gamma`, `m1 = sum(dxhat) / d`, `m2 = sum(dxhat * xhat) / d`, `dx = ((dxhat -
/// m1) - xhat * m2) / std`. `xhat` and `std` are `forward_batch`'s, or, with a 1D
/// `delta_batch` for one example, `forward`'s.
pub fn downstream_batch(delta_batch: &Array, gamma: &Array, xhat: &Array, std: &Array) -> Result<Array, Error> {
    let context = "layer_norm_downstream_batch";
    let d = gamma.data.len();
    let rows = feature_rows(delta_batch, d, "delta_batch", context)?;
    require_shape(xhat, delta_batch.shape, "xhat", context)?;
    require_shape(std, Shape::Vector(rows), "std", context)?;
    let mut dx = Vec::with_capacity(delta_batch.data.len());
    for ((delta, xh), &s) in delta_batch
        .data
        .chunks_exact(d)
        .zip(xhat.data.chunks_exact(d))
        .zip(std.data.iter())
    {
        let dxhat: Vec<f64> = delta.iter().zip(&gamma.data).map(|(&dv, &g)| dv * g).collect();
        let m1 = fold(dxhat.iter().copied()) / d as f64;
        let m2 = fold(dxhat.iter().zip(xh).map(|(&g, &x)| g * x)) / d as f64;
        dx.extend(dxhat.iter().zip(xh).map(|(&g, &x)| ((g - m1) - x * m2) / s));
    }
    Ok(Array {
        data: dx,
        shape: delta_batch.shape,
    })
}

/// `LayerNormArrayLayer.accumulate_gradient_batch`: `grad_gamma += sum(delta * xhat)` and
/// `grad_beta += sum(delta)`, each feature's sum over the rows (examples, then tokens). Returns the
/// updated `(grad_gamma, grad_beta)`. A 1D `delta_batch` and `xhat` are one example.
pub fn accumulate_gradient_batch(
    delta_batch: &Array,
    xhat: &Array,
    grad_gamma: &Array,
    grad_beta: &Array,
) -> Result<(Array, Array), Error> {
    let context = "layer_norm_accumulate_gradient_batch";
    let d = grad_gamma.data.len();
    feature_rows(delta_batch, d, "delta_batch", context)?;
    require_shape(xhat, delta_batch.shape, "xhat", context)?;
    require_shape(grad_beta, Shape::Vector(d), "grad_beta", context)?;
    let mut gamma_sum = vec![0.0; d];
    let mut beta_sum = vec![0.0; d];
    for (delta, xh) in delta_batch.data.chunks_exact(d).zip(xhat.data.chunks_exact(d)) {
        for j in 0..d {
            gamma_sum[j] += delta[j] * xh[j];
            beta_sum[j] += delta[j];
        }
    }
    let new_grad_gamma = grad_gamma.data.iter().zip(gamma_sum).map(|(&g, s)| g + s).collect();
    let new_grad_beta = grad_beta.data.iter().zip(beta_sum).map(|(&g, s)| g + s).collect();
    Ok((Array::from_vector(new_grad_gamma), Array::from_vector(new_grad_beta)))
}
