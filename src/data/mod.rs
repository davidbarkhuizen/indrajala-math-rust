//! The bindings of the core's `data`: dataset decoding, and inputs from small integer ids.

use pyo3::prelude::*;

pub(crate) mod mnist;
pub(crate) mod planes;

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    mnist::register(m)?;
    planes::register(m)?;
    Ok(())
}
