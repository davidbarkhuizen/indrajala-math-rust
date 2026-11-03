//! The core's `Error` as a Python exception.

use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

/// The core's error as the Python exception its kind names.
pub(crate) fn py_err(error: indrajala_math::Error) -> PyErr {
    match error {
        indrajala_math::Error::InvalidArgument(message) => PyValueError::new_err(message),
        indrajala_math::Error::Overflow(message) => PyOverflowError::new_err(message),
        _ => PyValueError::new_err(error.to_string()),
    }
}
