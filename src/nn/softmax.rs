//! The bindings of the core's `nn::softmax`: the softmax output layer.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::error::py_err;

/// `SoftmaxArrayLayer.forward`: `array_softmax(self.W @ x + self.b)`, `x`/`b` both 1D. Reuses
/// `array_softmax` (`ufuncs.rs`) directly rather than inlining its max/sum reduction, unlike
/// ReLU's trivial elementwise formula - the extra call is Rust-internal, not a second
/// Python/Rust FFI crossing.
#[pyfunction]
pub fn layer_softmax_forward(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::nn::softmax::forward(w, x, b)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `SoftmaxArrayLayer.forward_batch`: `array_softmax(X @ self.W.T + self.b)`, row-wise
/// normalization, `X` 2D (`batch, input_size`).
#[pyfunction]
pub fn layer_softmax_forward_batch(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    indrajala_math::nn::softmax::forward_batch(w, x, b)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `SoftmaxArrayLayer.compute_output_delta`/`compute_output_delta_batch`: `a - reference` -
/// `SoftmaxOutputNode`'s own simplification, no `a*(1-a)` damping term at all - shape-agnostic
/// like `layer_output_delta`, so one function covers both the single-example and batched case.
#[pyfunction]
pub fn layer_softmax_output_delta(a: &RustArray, reference: &RustArray) -> PyResult<RustArray> {
    indrajala_math::nn::softmax::output_delta(a, reference)
        .map(RustArray::from)
        .map_err(py_err)
}

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(layer_softmax_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layer_softmax_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_softmax_output_delta, m)?)?;
    Ok(())
}
