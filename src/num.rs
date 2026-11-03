//! The core's `num::activation` for the bindings: an activation by its Python name.

use pyo3::prelude::*;

use indrajala_math::num::activation::Activation;

use crate::error::py_err;

/// The activation by name (`"relu"`, `"gelu"`, `"gelu_tanh"`, `"silu"`), a `ValueError` otherwise.
pub(crate) fn activation_named(name: &str) -> PyResult<Activation> {
    Activation::from_name(name).map_err(py_err)
}
