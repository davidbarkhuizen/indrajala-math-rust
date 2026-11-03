//! The bindings of the core's `nn::optim`: the optimizers' parameter updates.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::error::py_err;

/// `ArrayLayer.apply_accumulated_gradient`: minibatch SGD in Goyal et al. 2017's grouping, eq. (2),
/// the mean gradient first and then the rate: `self.W -= learning_rate * (self._grad_W /
/// batch_size); self.b -= learning_rate * (self._grad_b / batch_size)` - shape-agnostic
/// (`W`/`grad_W` are always the same shape as each other, likewise `b`/`grad_b`), so one function
/// covers both the single-example (`batch_size=1`) and batched caller.
#[pyfunction]
pub fn layer_apply_accumulated_gradient(
    w: &RustArray,
    b: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    learning_rate: f64,
    batch_size: usize,
) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::nn::optim::apply_accumulated_gradient(w, b, grad_w, grad_b, learning_rate, batch_size)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

/// One single-example plain-SGD step: `layer_accumulate_gradient` into fresh zero accumulators,
/// then `layer_apply_accumulated_gradient` with `batch_size=1`, in one pass and one allocation
/// per parameter. `RustArrayNetworkBase.learn` runs exactly that pair on every layer, and apply
/// resets the accumulators, so at batch size 1 they are always fresh. Computes `w - lr * (0.0 +
/// delta[i] * x[j])`: the `0.0 +` is the add into the zero accumulator, kept on purpose. It turns
/// a `-0.0` product into `+0.0`, which changes the result where `w` is `-0.0`, so dropping it
/// would break bit-identity with the pair it replaces.
#[pyfunction]
pub fn layer_sgd_step(
    w: &RustArray,
    b: &RustArray,
    delta: &RustArray,
    input_activation: &RustArray,
    learning_rate: f64,
) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::nn::optim::sgd_step(w, b, delta, input_activation, learning_rate)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (w, b, grad_w, grad_b, m_w, v_w, m_b, v_b, t, beta1, beta2, epsilon, learning_rate, batch_size, weight_decay=0.0))]
pub fn layer_adam_apply_accumulated_gradient(
    w: &RustArray,
    b: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    m_w: &RustArray,
    v_w: &RustArray,
    m_b: &RustArray,
    v_b: &RustArray,
    t: u32,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    learning_rate: f64,
    batch_size: usize,
    weight_decay: f64,
) -> PyResult<(RustArray, RustArray, RustArray, RustArray, RustArray, RustArray)> {
    indrajala_math::nn::optim::adam_apply_accumulated_gradient(
        w,
        b,
        grad_w,
        grad_b,
        m_w,
        v_w,
        m_b,
        v_b,
        t,
        beta1,
        beta2,
        epsilon,
        learning_rate,
        batch_size,
        weight_decay,
    )
    .map(|r| {
        let (r0, r1, r2, r3, r4, r5) = r;
        (
            RustArray::from(r0),
            RustArray::from(r1),
            RustArray::from(r2),
            RustArray::from(r3),
            RustArray::from(r4),
            RustArray::from(r5),
        )
    })
    .map_err(py_err)
}

/// `L2ArrayLayer.apply_accumulated_gradient`: weight decay as Goyal et al. 2017's eq. (8), `λW`
/// added to the mean gradient: `W -= learning_rate * (grad_W / batch_size + l2_lambda * W);
/// b -= learning_rate * (grad_b / batch_size)` (bias unregularized). No persistent
/// per-parameter state at all (unlike
/// `layer_adam_apply_accumulated_gradient`/`layer_momentum_apply_accumulated_gradient`), so this
/// takes only `W`/`b`/`grad_W`/`grad_b` plus the scalar `l2_lambda` - the simplest fused op in
/// this round.
#[pyfunction]
pub fn layer_l2_apply_accumulated_gradient(
    w: &RustArray,
    b: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    l2_lambda: f64,
    learning_rate: f64,
    batch_size: usize,
) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::nn::optim::l2_apply_accumulated_gradient(w, b, grad_w, grad_b, l2_lambda, learning_rate, batch_size)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

/// `MomentumArrayLayer.apply_accumulated_gradient`: momentum as Goyal et al. 2017's eq. (9),
/// `velocity = momentum * velocity + grad / batch_size; param -= learning_rate * velocity`. It
/// replaces Rumelhart et al.'s eq. (10), which folds the rate into the velocity and so needs a
/// correction when the rate changes. One velocity array per parameter tensor, shape-agnostic like
/// `layer_apply_accumulated_gradient` (covers both the `W`/`grad_W`/`velocity_W` (2D) and
/// `b`/`grad_b`/`velocity_b` (1D) cases via two calls from the Python caller, the same convention
/// `layer_apply_accumulated_gradient` itself uses).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn layer_momentum_apply_accumulated_gradient(
    w: &RustArray,
    b: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    velocity_w: &RustArray,
    velocity_b: &RustArray,
    momentum: f64,
    learning_rate: f64,
    batch_size: usize,
) -> PyResult<(RustArray, RustArray, RustArray, RustArray)> {
    indrajala_math::nn::optim::momentum_apply_accumulated_gradient(
        w,
        b,
        grad_w,
        grad_b,
        velocity_w,
        velocity_b,
        momentum,
        learning_rate,
        batch_size,
    )
    .map(|r| {
        let (r0, r1, r2, r3) = r;
        (
            RustArray::from(r0),
            RustArray::from(r1),
            RustArray::from(r2),
            RustArray::from(r3),
        )
    })
    .map_err(py_err)
}

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(layer_apply_accumulated_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(layer_sgd_step, m)?)?;
    m.add_function(wrap_pyfunction!(layer_adam_apply_accumulated_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(layer_l2_apply_accumulated_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(layer_momentum_apply_accumulated_gradient, m)?)?;
    Ok(())
}
