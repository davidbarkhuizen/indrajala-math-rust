//! The bindings of the core's `nn::norm`: batch norm and layer norm.

use pyo3::prelude::*;

pub(crate) mod batch_norm;
pub(crate) mod layer_norm;

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    batch_norm::register(m)?;
    layer_norm::register(m)?;
    Ok(())
}
