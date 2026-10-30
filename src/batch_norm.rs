//! The bindings of the core's `batch_norm` (`indrajala_math::batch_norm`): each `#[pyfunction]` lends its
//! arrays to the core's and moves the results in.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::linalg::py_err;
#[allow(unused_imports)]
pub(crate) use indrajala_math::batch_norm::*;

/// `LinearArrayLayer.forward`: `self.W @ x`, `x` 1D, `linear_preactivation` without the bias.
#[pyfunction]
pub fn linear_forward(w: &RustArray, x: &RustArray) -> PyResult<RustArray> {
    indrajala_math::batch_norm::linear_forward(w, x)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `LinearArrayLayer.forward_batch`: `X @ self.W.T`, `linear_preactivation_batch` without the
/// bias, so row `i` is bit-identical to `linear_forward(w, X[i])`.
#[pyfunction]
pub fn linear_forward_batch(w: &RustArray, x: &RustArray) -> PyResult<RustArray> {
    indrajala_math::batch_norm::linear_forward_batch(w, x)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `LinearArrayLayer.accumulate_gradient_batch`: `self.grad_W += self.delta_batch.T @
/// input_activation_batch`, `layer_accumulate_gradient_batch`'s `grad_W` without its `grad_b`.
#[pyfunction]
pub fn linear_accumulate_gradient_batch(
    delta_batch: &RustArray,
    input_activation_batch: &RustArray,
    grad_w: &RustArray,
) -> PyResult<RustArray> {
    indrajala_math::batch_norm::linear_accumulate_gradient_batch(delta_batch, input_activation_batch, grad_w)
        .map(RustArray::from)
        .map_err(py_err)
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
    indrajala_math::batch_norm::batch_norm_forward(
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        epsilon,
        activation,
        positions,
    )
    .map(RustArray::from)
    .map_err(py_err)
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
#[pyfunction]
#[pyo3(signature = (x, gamma, beta, running_mean, running_var, epsilon, running_rate, activation, positions=1, group_size=None))]
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
    group_size: Option<usize>,
) -> PyResult<(
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
)> {
    indrajala_math::batch_norm::batch_norm_forward_batch(
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        epsilon,
        running_rate,
        activation,
        positions,
        group_size,
    )
    .map(|r| {
        let (r0, r1, r2, r3, r4, r5, r6) = r;
        (
            RustArray::from(r0),
            RustArray::from(r1),
            RustArray::from(r2),
            RustArray::from(r3),
            RustArray::from(r4),
            RustArray::from(r5),
            RustArray::from(r6),
        )
    })
    .map_err(py_err)
}

/// `BatchNormArrayLayer.downstream_batch`: `dl/dx`, the linear layer's delta, from `delta_batch`
/// (`dl/dy`, the activation's derivative already applied) by the paper's § 3 chain rule, term by
/// term, each ghost group over its own values. `d`, `var` and `std` are
/// `batch_norm_forward_batch`'s, with the same `positions` and `group_size`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (delta_batch, gamma, d, var, std, epsilon, positions=1, group_size=None))]
pub fn batch_norm_downstream_batch(
    delta_batch: &RustArray,
    gamma: &RustArray,
    d: &RustArray,
    var: &RustArray,
    std: &RustArray,
    epsilon: f64,
    positions: usize,
    group_size: Option<usize>,
) -> PyResult<RustArray> {
    indrajala_math::batch_norm::batch_norm_downstream_batch(
        delta_batch,
        gamma,
        d,
        var,
        std,
        epsilon,
        positions,
        group_size,
    )
    .map(RustArray::from)
    .map_err(py_err)
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
    indrajala_math::batch_norm::batch_norm_accumulate_gradient_batch(
        delta_batch,
        xhat,
        grad_gamma,
        grad_beta,
        positions,
    )
    .map(|r| {
        let (r0, r1) = r;
        (RustArray::from(r0), RustArray::from(r1))
    })
    .map_err(py_err)
}
