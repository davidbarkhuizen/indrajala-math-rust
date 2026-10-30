//! The core's products (`indrajala_math::linalg`) for the bindings: each wrapper lends the core
//! array inside each `Array` and moves the result in, so nothing is copied at the boundary.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use indrajala_math::linalg;

use crate::array::RustArray;

/// The core's error as the Python exception its kind names.
pub(crate) fn py_err(error: indrajala_math::Error) -> PyErr {
    match error {
        indrajala_math::Error::InvalidArgument(message) => PyValueError::new_err(message),
        _ => PyValueError::new_err(error.to_string()),
    }
}

/// `a @ b` (`indrajala_math::linalg::matmul`).
pub(crate) fn matmul(a: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    linalg::matmul(a, b).map(RustArray::from).map_err(py_err)
}

/// `c + a @ b`, bit-identical to the product then the add (`linalg::matmul_add`).
pub(crate) fn matmul_add(c: &RustArray, a: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    linalg::matmul_add(c, a, b).map(RustArray::from).map_err(py_err)
}

/// `a @ b.T` (`linalg::matmul_nt`).
pub(crate) fn matmul_nt(a: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    linalg::matmul_nt(a, b).map(RustArray::from).map_err(py_err)
}

/// Test/benchmark hook: max threads and threshold in flops for every threaded matmul
/// (`linalg::set_matmul_threading`; `(0, 0)` resets).
#[pyfunction]
pub(crate) fn set_matmul_threading(max_threads: usize, threshold_flops: usize) {
    linalg::set_matmul_threading(max_threads, threshold_flops);
}

/// Test hook: the thread count the threaded matmuls would use for an `m x k @ k x n` product.
#[pyfunction]
pub(crate) fn matmul_threads_for(m: usize, k: usize, n: usize) -> usize {
    linalg::matmul_threads_for(m, k, n)
}

/// Benchmark hook: the conv downstream's rows per block and `matmul_long_k`'s slab rows
/// (`linalg::set_kernel_overrides`; `(0, 0)` resets).
#[pyfunction]
pub(crate) fn set_kernel_overrides(narrow_rows_per_block: usize, long_k_block: usize) {
    linalg::set_kernel_overrides(narrow_rows_per_block, long_k_block);
}

#[pymethods]
impl RustArray {
    /// Exposed as Python's `@` operator (`__matmul__`), not a free function - every real call
    /// site (`self.W @ x`, `X @ self.W.T`, ...) uses `@` syntax, so the Rust binding matches it
    /// rather than requiring a rewrite to a `matmul(a, b)` call style.
    fn __matmul__(&self, other: &RustArray) -> PyResult<RustArray> {
        matmul(self, other)
    }
}

/// The full pairwise-product matrix of two 1D vectors - `accumulate_gradient`'s own
/// `np.outer(delta, input_layer.a)`. Exposed as a free function (`indrajala_math_rust.outer(a, b)`),
/// matching `np.outer`'s own call style rather than an operator.
#[pyfunction]
pub fn outer(a: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    linalg::outer(a, b).map(RustArray::from).map_err(py_err)
}
