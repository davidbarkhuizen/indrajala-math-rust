use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::RustArray;

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
    if record_size == 0 {
        return Err(PyValueError::new_err("record_size must be at least 1"));
    }
    if data.len() % record_size != 0 {
        return Err(PyValueError::new_err(format!(
            "data length {} is not a multiple of record_size {}",
            data.len(),
            record_size
        )));
    }

    let record_count = data.len() / record_size;
    let pixel_count = record_size - 1;
    let mut pixels = Vec::with_capacity(record_count * pixel_count);
    for record in 0..record_count {
        let base = record * record_size;
        for pixel in &data[base..base + pixel_count] {
            pixels.push(*pixel as f64 / 255.0);
        }
    }
    Ok(RustArray::from_matrix(pixels, record_count, pixel_count))
}
