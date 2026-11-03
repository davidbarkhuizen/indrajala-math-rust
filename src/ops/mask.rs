//! The bindings of the core's `ops::mask`: the masks the backward passes apply.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::error::py_err;
use crate::num::activation_named;

/// `downstream * act'(cached)` - a hidden layer's delta from a downstream it was handed (before a
/// layer with no `W` to read), the fused `layer_activation_hidden_delta*` ops' own elementwise
/// step. `cached` is the layer's `a` for ReLU (the mask `a > 0`), its `z` for the GELUs;
/// `activation` is `"relu"`, `"gelu"` or `"gelu_tanh"`. Shape-agnostic.
#[pyfunction]
pub fn array_activation_mask(downstream: &RustArray, cached: &RustArray, activation: &str) -> PyResult<RustArray> {
    let activation = activation_named(activation)?;
    indrajala_math::ops::mask::activation(downstream, cached, activation)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `downstream * a * (1 - a)` - the sigmoid's derivative applied to a downstream that a dense
/// layer's fused `layer_hidden_delta*` can't compute, because the next layer has no `W` to read
/// (a `LayerNorm`, or a residual fork whose body starts with one: indrajala-ml's layer-norm and
/// attention workplan, D5). The fused ops' elementwise expression, in their grouping, so given the
/// same downstream it computes their bits. Shape-agnostic, as `array_activation_mask`.
#[pyfunction]
pub fn array_sigmoid_mask(downstream: &RustArray, a: &RustArray) -> PyResult<RustArray> {
    indrajala_math::ops::mask::sigmoid(downstream, a)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `downstream * base_activation * (1 - base_activation) * scale`, `scale = mask /
/// keep_probability` if `was_training` else `1.0` - the dropout layer's hidden delta from a
/// downstream it was handed, as `array_sigmoid_mask` is the sigmoid layer's (D5): the fused
/// `layer_dropout_hidden_delta*` ops' own elementwise expression. Shape-agnostic.
#[pyfunction]
pub fn array_dropout_mask(
    downstream: &RustArray,
    base_activation: &RustArray,
    mask: &RustArray,
    keep_probability: f64,
    was_training: bool,
) -> PyResult<RustArray> {
    indrajala_math::ops::mask::dropout(downstream, base_activation, mask, keep_probability, was_training)
        .map(RustArray::from)
        .map_err(py_err)
}

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(array_activation_mask, m)?)?;
    m.add_function(wrap_pyfunction!(array_sigmoid_mask, m)?)?;
    m.add_function(wrap_pyfunction!(array_dropout_mask, m)?)?;
    Ok(())
}
