//! The bindings of the core's `data`: dataset decoding.

use pyo3::prelude::*;

pub(crate) mod mnist;

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    mnist::register(m)?;
    Ok(())
}
