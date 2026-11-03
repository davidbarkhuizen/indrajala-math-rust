//! The bindings of the core's `nn`: one Python function per layer pass, under its flat Python name.

use pyo3::prelude::*;

pub(crate) mod attention;
pub(crate) mod conv;
pub(crate) mod dense;
pub(crate) mod dropout;
pub(crate) mod gated;
pub(crate) mod norm;
pub(crate) mod optim;
pub(crate) mod recurrent;
pub(crate) mod softmax;
pub(crate) mod tokens;

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dense::register(m)?;
    dropout::register(m)?;
    softmax::register(m)?;
    gated::register(m)?;
    optim::register(m)?;
    norm::register(m)?;
    conv::register(m)?;
    attention::register(m)?;
    tokens::register(m)?;
    recurrent::register(m)?;
    Ok(())
}
