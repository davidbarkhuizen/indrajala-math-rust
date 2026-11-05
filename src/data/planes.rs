//! The bindings of the core's `data::planes`.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::error::py_err;

/// The core's `one_hot_planes`: `ids` as bytes (`ndarray.tobytes()` of a `uint8` array, a byte a
/// cell), `cells` a example, to the `(n, channels * cells)` channel-major one-hot input.
#[pyfunction]
pub fn one_hot_planes(ids: &[u8], cells: usize, channels: usize) -> PyResult<RustArray> {
    indrajala_math::data::planes::one_hot_planes(ids, cells, channels)
        .map(RustArray::from)
        .map_err(py_err)
}

/// Adds this module's functions to the Python module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(one_hot_planes, m)?)?;
    Ok(())
}
