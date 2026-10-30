//! The bindings of the core's `layer_norm` (`indrajala_math::layer_norm`): each `#[pyfunction]` lends its
//! arrays to the core's and moves the results in.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::linalg::py_err;
#[allow(unused_imports)]
pub(crate) use indrajala_math::layer_norm::*;

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
    indrajala_math::layer_norm::layer_norm_forward(x, gamma, beta, epsilon)
        .map(|r| {
            let (r0, r1, r2) = r;
            (RustArray::from(r0), RustArray::from(r1), RustArray::from(r2))
        })
        .map_err(py_err)
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
    indrajala_math::layer_norm::layer_norm_forward_batch(x, gamma, beta, epsilon)
        .map(|r| {
            let (r0, r1, r2) = r;
            (RustArray::from(r0), RustArray::from(r1), RustArray::from(r2))
        })
        .map_err(py_err)
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
    indrajala_math::layer_norm::layer_norm_downstream_batch(delta_batch, gamma, xhat, std)
        .map(RustArray::from)
        .map_err(py_err)
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
    indrajala_math::layer_norm::layer_norm_accumulate_gradient_batch(delta_batch, xhat, grad_gamma, grad_beta)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}
