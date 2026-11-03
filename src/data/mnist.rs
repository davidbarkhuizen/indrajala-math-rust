//! The bindings of the core's `data::mnist`: each `#[pyfunction]` lends its arrays to the core's
//! and moves the results in.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::error::py_err;

/// A direct port of `load_mnist_dataset_as_array`'s own decode
/// (`np.frombuffer(data, dtype=np.uint8).reshape(n, RECORD_SIZE)[:, :-1].astype(np.float64) / 255.0`),
/// fused into one pass rather than built from this core's general reshape/slice/astype
/// primitives - nothing else in the interface subset needs a `uint8` array type, so there is no
/// reusable machinery to build this on top of, only this one decode to port directly.
/// `data` is a raw byte buffer of `record_size`-byte
/// records (784 pixel bytes + 1 label byte for real MNIST); returns pixels only, normalized to
/// `[0.0, 1.0]`, shape `(record_count, record_size - 1)` - labels are dropped here exactly as
/// `load_mnist_dataset_as_array` itself drops them.
#[pyfunction]
pub fn decode_mnist_pixels(data: &[u8], record_size: usize) -> PyResult<RustArray> {
    indrajala_math::data::mnist::decode_pixels(data, record_size)
        .map(RustArray::from)
        .map_err(py_err)
}

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode_mnist_pixels, m)?)?;
    Ok(())
}
