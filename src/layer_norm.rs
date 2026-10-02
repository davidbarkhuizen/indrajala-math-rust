//! Layer norm (indrajala-ml's layer-norm and attention workplan, D5): one Rust function per method
//! of `LayerNormArrayLayer` in `indrajala_ml/model/layer_norm_array_layer.py`, as `batch_norm.rs`
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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};

/// A left fold from `0.0` in index order.
fn fold(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, |acc, v| acc + v)
}

/// The number of `features`-wide rows in `x`, checked to be whole.
fn feature_rows(x: &RustArray, features: usize, name: &str, context: &str) -> PyResult<usize> {
    let n = x.data.len();
    if features == 0 || !n.is_multiple_of(features) {
        return Err(PyValueError::new_err(format!(
            "{context} requires {name} of whole tokens of {features} features (gamma's length), got shape {:?}",
            x.shape
        )));
    }
    Ok(n / features)
}

fn require_shape(array: &RustArray, shape: Shape, name: &str, context: &str) -> PyResult<()> {
    if array.shape != shape {
        return Err(PyValueError::new_err(format!(
            "{context} requires {name} of shape {shape:?}, got {:?}",
            array.shape
        )));
    }
    Ok(())
}

/// The forward pass over `x`'s rows: `(a, xhat, std)`, `a` and `xhat` in `x`'s shape and `std`
/// one value per row.
fn normalize(
    x: &RustArray,
    gamma: &RustArray,
    beta: &RustArray,
    epsilon: f64,
    context: &str,
) -> PyResult<(RustArray, RustArray, RustArray)> {
    let d = gamma.data.len();
    let rows = feature_rows(x, d, "x", context)?;
    require_shape(beta, Shape::Vector(d), "beta", context)?;
    let mut a = Vec::with_capacity(x.data.len());
    let mut xhat = Vec::with_capacity(x.data.len());
    let mut std = Vec::with_capacity(rows);
    for row in x.data.chunks_exact(d) {
        let mu = fold(row.iter().copied()) / d as f64;
        let c: Vec<f64> = row.iter().map(|&v| v - mu).collect();
        let var = fold(c.iter().map(|&v| v * v)) / d as f64;
        let s = (var + epsilon).sqrt();
        for (j, &cj) in c.iter().enumerate() {
            let xh = cj / s;
            xhat.push(xh);
            a.push(gamma.data[j] * xh + beta.data[j]);
        }
        std.push(s);
    }
    Ok((
        RustArray {
            data: a,
            shape: x.shape,
        },
        RustArray {
            data: xhat,
            shape: x.shape,
        },
        RustArray::from_vector(std),
    ))
}

/// `LayerNormArrayLayer.forward`: one example, `x` 1D of `T * d` values. Returns `(a, xhat, std)`,
/// what the backward pass reads: `xhat` in `x`'s layout and `std = sqrt(var + eps)` per token.
/// Training and inference alike.
#[pyfunction]
pub fn layer_norm_forward(
    x: &RustArray,
    gamma: &RustArray,
    beta: &RustArray,
    epsilon: f64,
) -> PyResult<(RustArray, RustArray, RustArray)> {
    let context = "layer_norm_forward";
    if !matches!(x.shape, Shape::Vector(_)) {
        return Err(PyValueError::new_err(format!(
            "{context} requires a 1D x, got shape {:?}",
            x.shape
        )));
    }
    normalize(x, gamma, beta, epsilon, context)
}

/// `LayerNormArrayLayer.forward_batch`: `x` 2D (`batch, T * d`), each row `layer_norm_forward`'s
/// bits. Returns `(a, xhat, std)`, `std` per (example, token), 1D of `batch * T`.
#[pyfunction]
pub fn layer_norm_forward_batch(
    x: &RustArray,
    gamma: &RustArray,
    beta: &RustArray,
    epsilon: f64,
) -> PyResult<(RustArray, RustArray, RustArray)> {
    let context = "layer_norm_forward_batch";
    if !matches!(x.shape, Shape::Matrix(_, _)) {
        return Err(PyValueError::new_err(format!(
            "{context} requires a 2D x, got shape {:?}",
            x.shape
        )));
    }
    normalize(x, gamma, beta, epsilon, context)
}

/// `LayerNormArrayLayer.downstream_batch`: `dl/dx` from `delta_batch` (`dl/dy`), per row,
/// `dxhat = delta * gamma`, `m1 = sum(dxhat) / d`, `m2 = sum(dxhat * xhat) / d`, `dx = ((dxhat -
/// m1) - xhat * m2) / std`. `xhat` and `std` are `layer_norm_forward_batch`'s, or, with a 1D
/// `delta_batch` for one example, `layer_norm_forward`'s.
#[pyfunction]
pub fn layer_norm_downstream_batch(
    delta_batch: &RustArray,
    gamma: &RustArray,
    xhat: &RustArray,
    std: &RustArray,
) -> PyResult<RustArray> {
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
    Ok(RustArray {
        data: dx,
        shape: delta_batch.shape,
    })
}

/// `LayerNormArrayLayer.accumulate_gradient_batch`: `grad_gamma += sum(delta * xhat)` and
/// `grad_beta += sum(delta)`, each feature's sum over the rows (examples, then tokens). Returns the
/// updated `(grad_gamma, grad_beta)`. A 1D `delta_batch` and `xhat` are one example.
#[pyfunction]
pub fn layer_norm_accumulate_gradient_batch(
    delta_batch: &RustArray,
    xhat: &RustArray,
    grad_gamma: &RustArray,
    grad_beta: &RustArray,
) -> PyResult<(RustArray, RustArray)> {
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
    Ok((
        RustArray::from_vector(new_grad_gamma),
        RustArray::from_vector(new_grad_beta),
    ))
}
