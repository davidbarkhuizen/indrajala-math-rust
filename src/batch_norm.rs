//! Batch normalization (Ioffe & Szegedy 2015) and the bias-free linear layer before a dense one:
//! one Rust function per method of indrajala-ml's `indrajala_ml/model/batch_norm_array_layer.py`
//! and `linear_array_layer.py`, as `fused.rs` is for `array_layer.py`. (The bias-free conv layer's
//! ops are `conv.rs`'s `conv_linear_*`.)
//!
//! Each batch-norm function normalizes every channel over the batch. After a dense layer a
//! channel is a feature, a column of the `(N, C)` batch, and `positions` is 1. After a conv layer
//! the batch is its channel-major `(N, C*P)`, `positions` is `P`, and a channel's values are its
//! `P` positions in every example: `BatchNormArrayLayer`'s `(N*P, C)` view, which the functions
//! never build. They index the channel-major layout in place, example by example, channel by
//! channel, position by position, so each channel still meets its values in the view's row order,
//! the README's (example, position) order. At `positions = 1` that is the dense layer's row order.
//! `xhat` and `d` come back in the input's layout, as the numpy layer's `_flat` would give them.
//!
//! The batch-norm functions compute indrajala-ml's README expressions (Batch normalization), in
//! their grouping, left to right, with only IEEE 754's correctly rounded `+ - * /` and `sqrt`: no
//! `mul_add` and no `powf`. Every sum over the batch is a left fold from `0.0` in that order, as
//! `sum_axis0` is for a dense batch. So given the same inputs they compute numpy's bits.

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

/// A batch's example count: a 2D array `channels * positions` wide.
fn batch_rows(x: &RustArray, channels: usize, positions: usize, context: &str) -> PyResult<usize> {
    match x.shape {
        Shape::Matrix(rows, n) if n == channels * positions => Ok(rows),
        shape => Err(PyValueError::new_err(format!(
            "{context} requires a 2D batch of {channels} channels x {positions} positions, got shape {shape:?}"
        ))),
    }
}

/// The per-channel parameters and statistics: each 1D, one value per channel.
fn require_channels(arrays: &[(&RustArray, &str)], channels: usize, context: &str) -> PyResult<()> {
    for (array, name) in arrays {
        if array.shape != Shape::Vector(channels) {
            return Err(PyValueError::new_err(format!(
                "{context} requires {name} of shape {:?}, got {:?}",
                Shape::Vector(channels),
                array.shape
            )));
        }
    }
    Ok(())
}

fn require_positions(positions: usize, context: &str) -> PyResult<()> {
    if positions < 1 {
        return Err(PyValueError::new_err(format!(
            "{context} requires positions of 1 or more, got 0"
        )));
    }
    Ok(())
}

/// Each channel's sum over its values, a left fold from `0.0` in (example, position) order: for a
/// dense batch (`positions = 1`), `sum_axis0`'s row order.
fn sum_channels(data: &[f64], channels: usize, positions: usize) -> Vec<f64> {
    let mut out = vec![0.0; channels];
    for example in data.chunks_exact((channels * positions).max(1)) {
        for (acc, channel) in out.iter_mut().zip(example.chunks_exact(positions)) {
            for &v in channel {
                *acc += v;
            }
        }
    }
    out
}

/// Each value of a channel-major batch through `f(value, channel)`.
fn map_channels(data: &[f64], channels: usize, positions: usize, f: impl Fn(f64, usize) -> f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(data.len());
    for example in data.chunks_exact((channels * positions).max(1)) {
        for (j, channel) in example.chunks_exact(positions).enumerate() {
            out.extend(channel.iter().map(|&v| f(v, j)));
        }
    }
    out
}

/// Each pair of values of two same-shaped channel-major batches through `f(u, v, channel)`.
fn zip_channels(
    u: &[f64],
    v: &[f64],
    channels: usize,
    positions: usize,
    f: impl Fn(f64, f64, usize) -> f64,
) -> Vec<f64> {
    let width = (channels * positions).max(1);
    let mut out = Vec::with_capacity(u.len());
    for (u_example, v_example) in u.chunks_exact(width).zip(v.chunks_exact(width)) {
        for (j, (u_channel, v_channel)) in u_example
            .chunks_exact(positions)
            .zip(v_example.chunks_exact(positions))
            .enumerate()
        {
            out.extend(u_channel.iter().zip(v_channel).map(|(&a, &b)| f(a, b, j)));
        }
    }
    out
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

/// `BatchNormArrayLayer._inference`, a single example (`x` 1D) or a batch (`x` 2D), each
/// `channels * positions` values: `xhat = (x - running_mean) / sqrt(running_var + eps)`, then
/// `gamma * xhat + beta` and the activation, per channel. The running averages don't move.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (x, gamma, beta, running_mean, running_var, epsilon, activation, positions=1))]
pub fn batch_norm_forward(
    x: &RustArray,
    gamma: &RustArray,
    beta: &RustArray,
    running_mean: &RustArray,
    running_var: &RustArray,
    epsilon: f64,
    activation: &str,
    positions: usize,
) -> PyResult<RustArray> {
    let context = "batch_norm_forward";
    let activation = Activation::parse(activation, context)?;
    require_positions(positions, context)?;
    let channels = gamma.data.len();
    let width = match x.shape {
        Shape::Vector(n) | Shape::Matrix(_, n) => n,
    };
    if width != channels * positions {
        return Err(PyValueError::new_err(format!(
            "{context} requires x of {channels} channels x {positions} positions per example (gamma's length), \
             got shape {:?}",
            x.shape
        )));
    }
    require_channels(
        &[
            (beta, "beta"),
            (running_mean, "running_mean"),
            (running_var, "running_var"),
        ],
        channels,
        context,
    )?;
    let std: Vec<f64> = running_var.data.iter().map(|&v| (v + epsilon).sqrt()).collect();
    let data = map_channels(&x.data, channels, positions, |v, j| {
        let xhat = (v - running_mean.data[j]) / std[j];
        activation.apply(gamma.data[j] * xhat + beta.data[j])
    });
    Ok(RustArray { data, shape: x.shape })
}

/// `BatchNormArrayLayer.forward_batch` in training, `x` 2D (`batch, channels * positions`), batch
/// 2 or more: Algorithm 1 with the batch's statistics, `m = batch * positions` values per
/// channel, then the activation, and the running averages moved.
///
/// Returns `(a, xhat, d, var, std, running_mean, running_var)`: the activations, what the
/// backward pass reads (`xhat` and `d = x - mu` in `x`'s layout, `var` and `std = sqrt(var +
/// eps)` per channel), and the new running averages, the running variance taking the unbiased
/// `ss / (m - 1)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (x, gamma, beta, running_mean, running_var, epsilon, running_rate, activation, positions=1))]
pub fn batch_norm_forward_batch(
    x: &RustArray,
    gamma: &RustArray,
    beta: &RustArray,
    running_mean: &RustArray,
    running_var: &RustArray,
    epsilon: f64,
    running_rate: f64,
    activation: &str,
    positions: usize,
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
    require_positions(positions, context)?;
    let channels = gamma.data.len();
    let rows = batch_rows(x, channels, positions, context)?;
    require_channels(
        &[
            (beta, "beta"),
            (running_mean, "running_mean"),
            (running_var, "running_var"),
        ],
        channels,
        context,
    )?;
    if rows < 2 {
        return Err(PyValueError::new_err(format!(
            "{context} requires a batch of 2 or more in training, got {rows}"
        )));
    }
    let (c, p) = (channels, positions);
    let m = (rows * p) as f64;

    let mu: Vec<f64> = sum_channels(&x.data, c, p).iter().map(|&s| s / m).collect();
    let d = map_channels(&x.data, c, p, |v, j| v - mu[j]);
    let ss = sum_channels(&d.iter().map(|&v| v * v).collect::<Vec<f64>>(), c, p);
    let var: Vec<f64> = ss.iter().map(|&s| s / m).collect();
    let std: Vec<f64> = var.iter().map(|&v| (v + epsilon).sqrt()).collect();
    let xhat = map_channels(&d, c, p, |v, j| v / std[j]);
    let a = map_channels(&xhat, c, p, |v, j| activation.apply(gamma.data[j] * v + beta.data[j]));

    let new_running_mean: Vec<f64> = (0..c)
        .map(|j| (1.0 - running_rate) * running_mean.data[j] + running_rate * mu[j])
        .collect();
    let new_running_var: Vec<f64> = (0..c)
        .map(|j| (1.0 - running_rate) * running_var.data[j] + running_rate * (ss[j] / (m - 1.0)))
        .collect();

    let width = c * p;
    Ok((
        RustArray::from_matrix(a, rows, width),
        RustArray::from_matrix(xhat, rows, width),
        RustArray::from_matrix(d, rows, width),
        RustArray::from_vector(var),
        RustArray::from_vector(std),
        RustArray::from_vector(new_running_mean),
        RustArray::from_vector(new_running_var),
    ))
}

/// `BatchNormArrayLayer.downstream_batch`: `dl/dx`, the linear layer's delta, from `delta_batch`
/// (`dl/dy`, the activation's derivative already applied) by the paper's § 3 chain rule, term by
/// term. `d`, `var` and `std` are `batch_norm_forward_batch`'s, with the same `positions`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (delta_batch, gamma, d, var, std, epsilon, positions=1))]
pub fn batch_norm_downstream_batch(
    delta_batch: &RustArray,
    gamma: &RustArray,
    d: &RustArray,
    var: &RustArray,
    std: &RustArray,
    epsilon: f64,
    positions: usize,
) -> PyResult<RustArray> {
    let context = "batch_norm_downstream_batch";
    require_positions(positions, context)?;
    let channels = gamma.data.len();
    let rows = batch_rows(delta_batch, channels, positions, context)?;
    require_channels(&[(var, "var"), (std, "std")], channels, context)?;
    if d.shape != delta_batch.shape {
        return Err(PyValueError::new_err(format!(
            "{context} requires d of shape {:?}, got {:?}",
            delta_batch.shape, d.shape
        )));
    }
    let (c, p) = (channels, positions);
    let m = (rows * p) as f64;

    let dxhat = map_channels(&delta_batch.data, c, p, |v, j| v * gamma.data[j]);
    let inv_std: Vec<f64> = std.data.iter().map(|&s| 1.0 / s).collect();
    let inv_std3: Vec<f64> = (0..c).map(|j| inv_std[j] / (var.data[j] + epsilon)).collect();
    let dvar_terms = zip_channels(&dxhat, &d.data, c, p, |g, dv, j| g * dv * -0.5 * inv_std3[j]);
    let dvar = sum_channels(&dvar_terms, c, p);
    let dmu_terms = map_channels(&dxhat, c, p, |g, j| g * -inv_std[j]);
    let d_terms: Vec<f64> = d.data.iter().map(|&dv| -2.0 * dv).collect();
    let dmu_sum = sum_channels(&dmu_terms, c, p);
    let d_sum = sum_channels(&d_terms, c, p);
    let dmu: Vec<f64> = (0..c).map(|j| dmu_sum[j] + dvar[j] * d_sum[j] / m).collect();

    let dx = zip_channels(&dxhat, &d.data, c, p, |g, dv, j| {
        g * inv_std[j] + dvar[j] * (2.0 * dv) / m + dmu[j] / m
    });
    Ok(RustArray::from_matrix(dx, rows, c * p))
}

/// `BatchNormArrayLayer.accumulate_gradient_batch`: `grad_gamma += sum(delta * xhat)` and
/// `grad_beta += sum(delta)` over each channel's values. Returns the updated `(grad_gamma,
/// grad_beta)`.
#[pyfunction]
#[pyo3(signature = (delta_batch, xhat, grad_gamma, grad_beta, positions=1))]
pub fn batch_norm_accumulate_gradient_batch(
    delta_batch: &RustArray,
    xhat: &RustArray,
    grad_gamma: &RustArray,
    grad_beta: &RustArray,
    positions: usize,
) -> PyResult<(RustArray, RustArray)> {
    let context = "batch_norm_accumulate_gradient_batch";
    require_positions(positions, context)?;
    let channels = grad_gamma.data.len();
    batch_rows(delta_batch, channels, positions, context)?;
    require_channels(&[(grad_beta, "grad_beta")], channels, context)?;
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
    let gamma_sum = sum_channels(&products, channels, positions);
    let beta_sum = sum_channels(&delta_batch.data, channels, positions);
    let new_grad_gamma = grad_gamma.data.iter().zip(gamma_sum).map(|(&g, s)| g + s).collect();
    let new_grad_beta = grad_beta.data.iter().zip(beta_sum).map(|(&g, s)| g + s).collect();
    Ok((
        RustArray::from_vector(new_grad_gamma),
        RustArray::from_vector(new_grad_beta),
    ))
}
