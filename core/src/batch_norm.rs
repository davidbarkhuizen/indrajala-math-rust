//! Batch normalization (Ioffe & Szegedy 2015) and the bias-free linear layer before a dense one:
//! one Rust function per method of indrajala-ml's `indrajala_ml/model/layers/numpy/batch_norm_array_layer.py`
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

use crate::array::{Array, Shape};
use crate::cr_exp::sigmoid_in_place;
use crate::linalg::{matmul, matmul_add, matmul_nt};
use crate::scalar::relu;
use crate::Error;

/// A batch-norm layer's activation, fused into it as into a dense layer.
#[derive(Clone, Copy)]
enum Activation {
    Sigmoid,
    Relu,
}

impl Activation {
    fn parse(name: &str, context: &str) -> Result<Self, Error> {
        match name {
            "sigmoid" => Ok(Activation::Sigmoid),
            "relu" => Ok(Activation::Relu),
            _ => Err(Error::invalid(format!(
                "{context} requires activation 'sigmoid' or 'relu', got {name:?}"
            ))),
        }
    }

    /// `fused.rs`'s sigmoid and ReLU, in place.
    fn apply(self, values: &mut [f64]) {
        match self {
            Activation::Sigmoid => sigmoid_in_place(values),
            Activation::Relu => values.iter_mut().for_each(|v| *v = relu(*v)),
        }
    }
}

/// A batch's example count: a 2D array `channels * positions` wide.
fn batch_rows(x: &Array, channels: usize, positions: usize, context: &str) -> Result<usize, Error> {
    match x.shape {
        Shape::Matrix(rows, n) if n == channels * positions => Ok(rows),
        shape => Err(Error::invalid(format!(
            "{context} requires a 2D batch of {channels} channels x {positions} positions, got shape {shape:?}"
        ))),
    }
}

/// The per-channel parameters and statistics: each 1D, one value per channel.
fn require_channels(arrays: &[(&Array, &str)], channels: usize, context: &str) -> Result<(), Error> {
    for (array, name) in arrays {
        if array.shape != Shape::Vector(channels) {
            return Err(Error::invalid(format!(
                "{context} requires {name} of shape {:?}, got {:?}",
                Shape::Vector(channels),
                array.shape
            )));
        }
    }
    Ok(())
}

fn require_positions(positions: usize, context: &str) -> Result<(), Error> {
    if positions < 1 {
        return Err(Error::invalid(format!(
            "{context} requires positions of 1 or more, got 0"
        )));
    }
    Ok(())
}

/// A training batch's ghost groups (Hoffer et al. 2017), as `(first, end)` example ranges: runs
/// of `group_size` examples in row order, the last one the remainder, or the whole batch when
/// `group_size` is `None`. Each group needs 2 or more examples, as a batch does.
fn groups(rows: usize, group_size: Option<usize>, context: &str) -> Result<Vec<(usize, usize)>, Error> {
    let size = match group_size {
        None => rows,
        Some(size) if size >= 2 => size,
        Some(size) => {
            return Err(Error::invalid(format!(
                "{context} requires a group_size of 2 or more, got {size}"
            )));
        }
    };
    let ranges: Vec<(usize, usize)> = (0..rows)
        .step_by(size.max(1))
        .map(|first| (first, (first + size).min(rows)))
        .collect();
    if let Some(&(first, end)) = ranges.last() {
        if end - first < 2 {
            return Err(Error::invalid(format!(
                "{context} requires every group of 2 or more examples, got a batch of {rows} in groups of {size}, the last of 1"
            )));
        }
    }
    Ok(ranges)
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
pub fn linear_forward(w: &Array, x: &Array) -> Result<Array, Error> {
    if !matches!(x.shape, Shape::Vector(_)) {
        return Err(Error::invalid(format!(
            "linear_forward requires a 1D x, got shape {:?}",
            x.shape
        )));
    }
    matmul(w, x)
}

/// `LinearArrayLayer.forward_batch`: `X @ self.W.T`, `linear_preactivation_batch` without the
/// bias, so row `i` is bit-identical to `linear_forward(w, X[i])`.
pub fn linear_forward_batch(w: &Array, x: &Array) -> Result<Array, Error> {
    matmul_nt(x, w)
}

/// `LinearArrayLayer.accumulate_gradient_batch`: `self.grad_W += self.delta_batch.T @
/// input_activation_batch`, `layer_accumulate_gradient_batch`'s `grad_W` without its `grad_b`.
pub fn linear_accumulate_gradient_batch(
    delta_batch: &Array,
    input_activation_batch: &Array,
    grad_w: &Array,
) -> Result<Array, Error> {
    matmul_add(grad_w, &delta_batch.transpose(), input_activation_batch)
}

/// `BatchNormArrayLayer._inference`, a single example (`x` 1D) or a batch (`x` 2D), each
/// `channels * positions` values: `xhat = (x - running_mean) / sqrt(running_var + eps)`, then
/// `gamma * xhat + beta` and the activation, per channel. The running averages don't move.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_forward(
    x: &Array,
    gamma: &Array,
    beta: &Array,
    running_mean: &Array,
    running_var: &Array,
    epsilon: f64,
    activation: &str,
    positions: usize,
) -> Result<Array, Error> {
    let context = "batch_norm_forward";
    let activation = Activation::parse(activation, context)?;
    require_positions(positions, context)?;
    let channels = gamma.data.len();
    let width = match x.shape {
        Shape::Vector(n) | Shape::Matrix(_, n) => n,
    };
    if width != channels * positions {
        return Err(Error::invalid(format!(
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
    let mut data = map_channels(&x.data, channels, positions, |v, j| {
        let xhat = (v - running_mean.data[j]) / std[j];
        gamma.data[j] * xhat + beta.data[j]
    });
    activation.apply(&mut data);
    Ok(Array { data, shape: x.shape })
}

/// `BatchNormArrayLayer.forward_batch` in training, `x` 2D (`batch, channels * positions`), batch
/// 2 or more: Algorithm 1 with the batch's statistics, `m = batch * positions` values per
/// channel, then the activation, and the running averages moved.
///
/// With a `group_size`, each ghost group (`groups`) is normalized with its own statistics, `m =
/// group rows * positions`, and moves the running averages in turn, in row order. Without one,
/// the batch is one group.
///
/// Returns `(a, xhat, d, var, std, running_mean, running_var)`: the activations, what the
/// backward pass reads (`xhat` and `d = x - mu` in `x`'s layout, `var` and `std = sqrt(var +
/// eps)` per group per channel, group-major, so 1D of `channels` for one group), and the new
/// running averages, the running variance taking each group's unbiased `ss / (m - 1)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn batch_norm_forward_batch(
    x: &Array,
    gamma: &Array,
    beta: &Array,
    running_mean: &Array,
    running_var: &Array,
    epsilon: f64,
    running_rate: f64,
    activation: &str,
    positions: usize,
    group_size: Option<usize>,
) -> Result<(Array, Array, Array, Array, Array, Array, Array), Error> {
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
        return Err(Error::invalid(format!(
            "{context} requires a batch of 2 or more in training, got {rows}"
        )));
    }
    let ranges = groups(rows, group_size, context)?;
    let (c, p) = (channels, positions);
    let width = c * p;

    let mut a = Vec::with_capacity(x.data.len());
    let mut xhat = Vec::with_capacity(x.data.len());
    let mut d = Vec::with_capacity(x.data.len());
    let mut var = Vec::with_capacity(ranges.len() * c);
    let mut std = Vec::with_capacity(ranges.len() * c);
    let mut new_running_mean = running_mean.data.clone();
    let mut new_running_var = running_var.data.clone();
    for &(first, end) in &ranges {
        let group = &x.data[first * width..end * width];
        let m = ((end - first) * p) as f64;

        let mu: Vec<f64> = sum_channels(group, c, p).iter().map(|&s| s / m).collect();
        let group_d = map_channels(group, c, p, |v, j| v - mu[j]);
        let ss = sum_channels(&group_d.iter().map(|&v| v * v).collect::<Vec<f64>>(), c, p);
        let group_var: Vec<f64> = ss.iter().map(|&s| s / m).collect();
        let group_std: Vec<f64> = group_var.iter().map(|&v| (v + epsilon).sqrt()).collect();
        let group_xhat = map_channels(&group_d, c, p, |v, j| v / group_std[j]);
        let mut group_a = map_channels(&group_xhat, c, p, |v, j| gamma.data[j] * v + beta.data[j]);
        activation.apply(&mut group_a);
        a.extend(group_a);

        for j in 0..c {
            new_running_mean[j] = (1.0 - running_rate) * new_running_mean[j] + running_rate * mu[j];
            new_running_var[j] = (1.0 - running_rate) * new_running_var[j] + running_rate * (ss[j] / (m - 1.0));
        }

        xhat.extend(group_xhat);
        d.extend(group_d);
        var.extend(group_var);
        std.extend(group_std);
    }

    Ok((
        Array::from_matrix(a, rows, width),
        Array::from_matrix(xhat, rows, width),
        Array::from_matrix(d, rows, width),
        Array::from_vector(var),
        Array::from_vector(std),
        Array::from_vector(new_running_mean),
        Array::from_vector(new_running_var),
    ))
}

/// `BatchNormArrayLayer.downstream_batch`: `dl/dx`, the linear layer's delta, from `delta_batch`
/// (`dl/dy`, the activation's derivative already applied) by the paper's § 3 chain rule, term by
/// term, each ghost group over its own values. `d`, `var` and `std` are
/// `batch_norm_forward_batch`'s, with the same `positions` and `group_size`.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_downstream_batch(
    delta_batch: &Array,
    gamma: &Array,
    d: &Array,
    var: &Array,
    std: &Array,
    epsilon: f64,
    positions: usize,
    group_size: Option<usize>,
) -> Result<Array, Error> {
    let context = "batch_norm_downstream_batch";
    require_positions(positions, context)?;
    let channels = gamma.data.len();
    let rows = batch_rows(delta_batch, channels, positions, context)?;
    let ranges = groups(rows, group_size, context)?;
    require_channels(&[(var, "var"), (std, "std")], ranges.len() * channels, context)?;
    if d.shape != delta_batch.shape {
        return Err(Error::invalid(format!(
            "{context} requires d of shape {:?}, got {:?}",
            delta_batch.shape, d.shape
        )));
    }
    let (c, p) = (channels, positions);
    let width = c * p;

    let mut dx = Vec::with_capacity(delta_batch.data.len());
    for (g, &(first, end)) in ranges.iter().enumerate() {
        let group_delta = &delta_batch.data[first * width..end * width];
        let group_d = &d.data[first * width..end * width];
        let group_var = &var.data[g * c..(g + 1) * c];
        let group_std = &std.data[g * c..(g + 1) * c];
        let m = ((end - first) * p) as f64;

        let dxhat = map_channels(group_delta, c, p, |v, j| v * gamma.data[j]);
        let inv_std: Vec<f64> = group_std.iter().map(|&s| 1.0 / s).collect();
        let inv_std3: Vec<f64> = (0..c).map(|j| inv_std[j] / (group_var[j] + epsilon)).collect();
        let dvar_terms = zip_channels(&dxhat, group_d, c, p, |g, dv, j| g * dv * -0.5 * inv_std3[j]);
        let dvar = sum_channels(&dvar_terms, c, p);
        let dmu_terms = map_channels(&dxhat, c, p, |g, j| g * -inv_std[j]);
        let d_terms: Vec<f64> = group_d.iter().map(|&dv| -2.0 * dv).collect();
        let dmu_sum = sum_channels(&dmu_terms, c, p);
        let d_sum = sum_channels(&d_terms, c, p);
        let dmu: Vec<f64> = (0..c).map(|j| dmu_sum[j] + dvar[j] * d_sum[j] / m).collect();

        dx.extend(zip_channels(&dxhat, group_d, c, p, |g, dv, j| {
            g * inv_std[j] + dvar[j] * (2.0 * dv) / m + dmu[j] / m
        }));
    }
    Ok(Array::from_matrix(dx, rows, width))
}

/// `BatchNormArrayLayer.accumulate_gradient_batch`: `grad_gamma += sum(delta * xhat)` and
/// `grad_beta += sum(delta)` over each channel's values. Returns the updated `(grad_gamma,
/// grad_beta)`.
pub fn batch_norm_accumulate_gradient_batch(
    delta_batch: &Array,
    xhat: &Array,
    grad_gamma: &Array,
    grad_beta: &Array,
    positions: usize,
) -> Result<(Array, Array), Error> {
    let context = "batch_norm_accumulate_gradient_batch";
    require_positions(positions, context)?;
    let channels = grad_gamma.data.len();
    batch_rows(delta_batch, channels, positions, context)?;
    require_channels(&[(grad_beta, "grad_beta")], channels, context)?;
    if xhat.shape != delta_batch.shape {
        return Err(Error::invalid(format!(
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
    Ok((Array::from_vector(new_grad_gamma), Array::from_vector(new_grad_beta)))
}
