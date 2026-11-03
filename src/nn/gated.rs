//! The bindings of the core's `nn::gated`: the gated token-wise layer.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::error::py_err;
use crate::num::activation_named;

/// A gated token-wise layer's forward on its rows (indrajala-ml's FFN activations workplan, D3):
/// `z = X @ W.T + b`, `W` `(2 * size, input_size)` the gate's rows then the value's, and
/// `act(a) * v` of each row's gate and value, `activation` `"silu"` (SwiGLU), `"gelu"` or
/// `"gelu_tanh"` (GEGLU). Returns `(out, z)`, `out` `(rows, size)` and `z` `(rows, 2 * size)`.
#[pyfunction]
pub fn gated_forward_batch(
    w: &RustArray,
    x: &RustArray,
    b: &RustArray,
    activation: &str,
) -> PyResult<(RustArray, RustArray)> {
    let activation = activation_named(activation)?;
    indrajala_math::nn::gated::forward_batch(w, x, b, activation)
        .map(|(out, z)| (RustArray::from(out), RustArray::from(z)))
        .map_err(py_err)
}

/// A gated token-wise layer's delta of its pre-activation from its output's `delta` `(rows,
/// size)`: each row's gate part `(d * v) * act'(a)` then its value part `d * act(a)`, `(rows, 2 *
/// size)` as `z`.
#[pyfunction]
pub fn gated_delta_batch(delta: &RustArray, z: &RustArray, activation: &str) -> PyResult<RustArray> {
    let activation = activation_named(activation)?;
    indrajala_math::nn::gated::delta_batch(delta, z, activation)
        .map(RustArray::from)
        .map_err(py_err)
}

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gated_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(gated_delta_batch, m)?)?;
    Ok(())
}
