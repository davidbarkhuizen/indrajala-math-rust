//! The bindings of the core's `random`: numpy's legacy global generator and `default_rng`.

use pyo3::prelude::*;

pub(crate) mod generator;
pub(crate) mod legacy;

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    legacy::register(m)?;
    generator::register(m)?;
    Ok(())
}
