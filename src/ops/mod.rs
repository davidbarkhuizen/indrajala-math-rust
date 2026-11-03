//! The bindings of the core's `ops`: elementwise arithmetic, functions of one array, and masks.

use pyo3::prelude::*;

pub(crate) mod arith;
pub(crate) mod mask;
pub(crate) mod unary;

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    unary::register(m)?;
    mask::register(m)?;
    Ok(())
}
