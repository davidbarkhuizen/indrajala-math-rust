//! Dense batch normalization (Ioffe & Szegedy 2015) and the bias-free linear layer before it: one
//! Rust function per method of indrajala-ml's `indrajala_ml/model/batch_norm_array_layer.py` and
//! `linear_array_layer.py`, as `fused.rs` is for `array_layer.py`. Each normalizes every feature
//! (column) over the batch (rows).
//!
//! The batch-norm functions compute indrajala-ml's README expressions (Batch normalization), in
//! their grouping, left to right, with only IEEE 754's correctly rounded `+ - * /` and `sqrt`: no
//! `mul_add` and no `powf`. Every sum over the batch is a left fold from `0.0` in row order, as
//! `sum_axis0`. So given the same inputs they compute numpy's bits.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};
use crate::linalg::{matmul, matmul_add, matmul_nt};

/// A batch-norm layer's activation, fused into it as into a dense layer.
#[derive(Clone, Copy)]
enum Activation {
    Sigmoid,
    Relu,
}

impl Activation {
    fn parse(name: &str, context: &str) -> PyResult<Self> {
        match name {
            "sigmoid" => Ok(Activation::Sigmoid),
            "relu" => Ok(Activation::Relu),
            _ => Err(PyValueError::new_err(format!(
                "{context} requires activation 'sigmoid' or 'relu', got {name:?}"
            ))),
        }
    }

    /// `fused.rs`'s sigmoid and ReLU, per value.
    fn apply(self, y: f64) -> f64 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-y).exp()),
            Activation::Relu => y.max(0.0),
        }
    }
}

/// `(rows, cols)` of a batch: a 2D array, `cols` wide.
fn batch_shape(x: &RustArray, cols: usize, context: &str) -> PyResult<(usize, usize)> {
    match x.shape {
        Shape::Matrix(rows, n) if n == cols => Ok((rows, n)),
        shape => Err(PyValueError::new_err(format!(
            "{context} requires a 2D batch of {cols} features, got shape {shape:?}"
        ))),
    }
}

/// The per-feature parameters and statistics: each 1D, one value per feature.
fn require_features(arrays: &[(&RustArray, &str)], cols: usize, context: &str) -> PyResult<()> {
    for (array, name) in arrays {
        if array.shape != Shape::Vector(cols) {
            return Err(PyValueError::new_err(format!(
                "{context} requires {name} of shape {:?}, got {:?}",
                Shape::Vector(cols),
                array.shape
            )));
        }
    }
    Ok(())
}

/// Each column's sum over the rows, a left fold from `0.0` in row order: `sum_axis0`'s order.
fn sum_rows(data: &[f64], cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; cols];
    for row in data.chunks_exact(cols.max(1)) {
        for (acc, &v) in out.iter_mut().zip(row.iter()) {
            *acc += v;
        }
    }
    out
}

/// Each value of a row-major `rows x cols` batch through `f(value, column)`.
fn map_columns(data: &[f64], cols: usize, f: impl Fn(f64, usize) -> f64) -> Vec<f64> {
    data.iter().enumerate().map(|(i, &v)| f(v, i % cols)).collect()
}

/// `LinearArrayLayer.forward`: `self.W @ x`, `x` 1D, `linear_preactivation` without the bias.
#[pyfunction]
pub fn linear_forward(w: &RustArray, x: &RustArray) -> PyResult<RustArray> {
    if !matches!(x.shape, Shape::Vector(_)) {
        return Err(PyValueError::new_err(format!(
            "linear_forward requires a 1D x, got shape {:?}",
            x.shape
        )));
    }
    matmul(w, x)
}

/// `LinearArrayLayer.forward_batch`: `X @ self.W.T`, `linear_preactivation_batch` without the
/// bias, so row `i` is bit-identical to `linear_forward(w, X[i])`.
#[pyfunction]
pub fn linear_forward_batch(w: &RustArray, x: &RustArray) -> PyResult<RustArray> {
    matmul_nt(x, w)
}

/// `LinearArrayLayer.accumulate_gradient_batch`: `self.grad_W += self.delta_batch.T @
/// input_activation_batch`, `layer_accumulate_gradient_batch`'s `grad_W` without its `grad_b`.
#[pyfunction]
pub fn linear_accumulate_gradient_batch(
    delta_batch: &RustArray,
    input_activation_batch: &RustArray,
    grad_w: &RustArray,
) -> PyResult<RustArray> {
    matmul_add(grad_w, &delta_batch.transpose(), input_activation_batch)
}

/// `BatchNormArrayLayer._inference`, a single example (`x` 1D) or a batch (`x` 2D): `xhat =
/// (x - running_mean) / sqrt(running_var + eps)`, then `gamma * xhat + beta` and the activation.
/// The running averages don't move.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn batch_norm_forward(
    x: &RustArray,
    gamma: &RustArray,
    beta: &RustArray,
    running_mean: &RustArray,
    running_var: &RustArray,
    epsilon: f64,
    activation: &str,
) -> PyResult<RustArray> {
    let context = "batch_norm_forward";
    let activation = Activation::parse(activation, context)?;
    let cols = match x.shape {
        Shape::Vector(n) | Shape::Matrix(_, n) => n,
    };
    require_features(
        &[
            (gamma, "gamma"),
            (beta, "beta"),
            (running_mean, "running_mean"),
            (running_var, "running_var"),
        ],
        cols,
        context,
    )?;
    let std: Vec<f64> = running_var.data.iter().map(|&v| (v + epsilon).sqrt()).collect();
    let data = map_columns(&x.data, cols.max(1), |v, j| {
        let xhat = (v - running_mean.data[j]) / std[j];
        activation.apply(gamma.data[j] * xhat + beta.data[j])
    });
    Ok(RustArray { data, shape: x.shape })
}

/// `BatchNormArrayLayer.forward_batch` in training, `x` 2D (`batch, features`), batch 2 or more:
/// Algorithm 1 with the batch's statistics, then the activation, and the running averages moved.
///
/// Returns `(a, xhat, d, var, std, running_mean, running_var)`: the activations, what the
/// backward pass reads (`xhat`, `d = x - mu`, `var` and `std = sqrt(var + eps)`, per feature), and
/// the new running averages, the running variance taking the unbiased `ss / (m - 1)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
pub fn batch_norm_forward_batch(
    x: &RustArray,
    gamma: &RustArray,
    beta: &RustArray,
    running_mean: &RustArray,
    running_var: &RustArray,
    epsilon: f64,
    running_rate: f64,
    activation: &str,
) -> PyResult<(
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
)> {
    let context = "batch_norm_forward_batch";
    let activation = Activation::parse(activation, context)?;
    let cols = gamma.data.len();
    let (rows, cols) = batch_shape(x, cols, context)?;
    require_features(
        &[
            (gamma, "gamma"),
            (beta, "beta"),
            (running_mean, "running_mean"),
            (running_var, "running_var"),
        ],
        cols,
        context,
    )?;
    if rows < 2 {
        return Err(PyValueError::new_err(format!(
            "{context} requires a batch of 2 or more in training, got {rows}"
        )));
    }
    let m = rows as f64;

    let mu: Vec<f64> = sum_rows(&x.data, cols).iter().map(|&s| s / m).collect();
    let d = map_columns(&x.data, cols, |v, j| v - mu[j]);
    let ss = sum_rows(&d.iter().map(|&v| v * v).collect::<Vec<f64>>(), cols);
    let var: Vec<f64> = ss.iter().map(|&s| s / m).collect();
    let std: Vec<f64> = var.iter().map(|&v| (v + epsilon).sqrt()).collect();
    let xhat = map_columns(&d, cols, |v, j| v / std[j]);
    let a = map_columns(&xhat, cols, |v, j| activation.apply(gamma.data[j] * v + beta.data[j]));

    let new_running_mean: Vec<f64> = (0..cols)
        .map(|j| (1.0 - running_rate) * running_mean.data[j] + running_rate * mu[j])
        .collect();
    let new_running_var: Vec<f64> = (0..cols)
        .map(|j| (1.0 - running_rate) * running_var.data[j] + running_rate * (ss[j] / (m - 1.0)))
        .collect();

    Ok((
        RustArray::from_matrix(a, rows, cols),
        RustArray::from_matrix(xhat, rows, cols),
        RustArray::from_matrix(d, rows, cols),
        RustArray::from_vector(var),
        RustArray::from_vector(std),
        RustArray::from_vector(new_running_mean),
        RustArray::from_vector(new_running_var),
    ))
}

/// `BatchNormArrayLayer.downstream_batch`: `dl/dx`, the linear layer's delta, from `delta_batch`
/// (`dl/dy`, the activation's derivative already applied) by the paper's § 3 chain rule, term by
/// term. `d`, `var` and `std` are `batch_norm_forward_batch`'s.
#[pyfunction]
pub fn batch_norm_downstream_batch(
    delta_batch: &RustArray,
    gamma: &RustArray,
    d: &RustArray,
    var: &RustArray,
    std: &RustArray,
    epsilon: f64,
) -> PyResult<RustArray> {
    let context = "batch_norm_downstream_batch";
    let cols = gamma.data.len();
    let (rows, cols) = batch_shape(delta_batch, cols, context)?;
    require_features(&[(gamma, "gamma"), (var, "var"), (std, "std")], cols, context)?;
    if d.shape != delta_batch.shape {
        return Err(PyValueError::new_err(format!(
            "{context} requires d of shape {:?}, got {:?}",
            delta_batch.shape, d.shape
        )));
    }
    let m = rows as f64;

    let dxhat = map_columns(&delta_batch.data, cols, |v, j| v * gamma.data[j]);
    let inv_std: Vec<f64> = std.data.iter().map(|&s| 1.0 / s).collect();
    let inv_std3: Vec<f64> = (0..cols).map(|j| inv_std[j] / (var.data[j] + epsilon)).collect();
    let dvar_terms: Vec<f64> = dxhat
        .iter()
        .zip(d.data.iter())
        .enumerate()
        .map(|(i, (&g, &dv))| g * dv * -0.5 * inv_std3[i % cols])
        .collect();
    let dvar = sum_rows(&dvar_terms, cols);
    let dmu_terms = map_columns(&dxhat, cols, |g, j| g * -inv_std[j]);
    let d_terms: Vec<f64> = d.data.iter().map(|&dv| -2.0 * dv).collect();
    let dmu_sum = sum_rows(&dmu_terms, cols);
    let d_sum = sum_rows(&d_terms, cols);
    let dmu: Vec<f64> = (0..cols).map(|j| dmu_sum[j] + dvar[j] * d_sum[j] / m).collect();

    let dx = dxhat
        .iter()
        .zip(d.data.iter())
        .enumerate()
        .map(|(i, (&g, &dv))| {
            let j = i % cols;
            g * inv_std[j] + dvar[j] * (2.0 * dv) / m + dmu[j] / m
        })
        .collect();
    Ok(RustArray::from_matrix(dx, rows, cols))
}

/// `BatchNormArrayLayer.accumulate_gradient_batch`: `grad_gamma += sum(delta * xhat)` and
/// `grad_beta += sum(delta)` over the batch. Returns the updated `(grad_gamma, grad_beta)`.
#[pyfunction]
pub fn batch_norm_accumulate_gradient_batch(
    delta_batch: &RustArray,
    xhat: &RustArray,
    grad_gamma: &RustArray,
    grad_beta: &RustArray,
) -> PyResult<(RustArray, RustArray)> {
    let context = "batch_norm_accumulate_gradient_batch";
    let cols = grad_gamma.data.len();
    batch_shape(delta_batch, cols, context)?;
    require_features(&[(grad_gamma, "grad_gamma"), (grad_beta, "grad_beta")], cols, context)?;
    if xhat.shape != delta_batch.shape {
        return Err(PyValueError::new_err(format!(
            "{context} requires xhat of shape {:?}, got {:?}",
            delta_batch.shape, xhat.shape
        )));
    }
    let products: Vec<f64> = delta_batch
        .data
        .iter()
        .zip(xhat.data.iter())
        .map(|(&dv, &xv)| dv * xv)
        .collect();
    let gamma_sum = sum_rows(&products, cols);
    let beta_sum = sum_rows(&delta_batch.data, cols);
    let new_grad_gamma = grad_gamma.data.iter().zip(gamma_sum).map(|(&g, s)| g + s).collect();
    let new_grad_beta = grad_beta.data.iter().zip(beta_sum).map(|(&g, s)| g + s).collect();
    Ok((
        RustArray::from_vector(new_grad_gamma),
        RustArray::from_vector(new_grad_beta),
    ))
}
