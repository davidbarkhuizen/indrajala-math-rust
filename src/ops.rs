use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::array::RustArray;
use crate::linalg::py_err;
use indrajala_math::Array;

/// Accepts a Python int or float as a scalar operand (`grad_W / batch_size` passes a Python
/// int) - numpy itself accepts either for a scalar operation, so this core does too.
fn extract_scalar(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(v) = value.extract::<f64>() {
        return Ok(v);
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok(v as f64);
    }
    Err(PyTypeError::new_err("expected a numeric scalar"))
}

pub(crate) use indrajala_math::ops::{same_shape_elementwise, scalar_elementwise};

impl RustArray {
    /// The core's `combine_with_array`, raised as `ValueError`; for the bindings' own modules.
    pub(crate) fn combine_with_array(
        &self,
        other: &RustArray,
        op: impl Fn(f64, f64) -> f64,
        op_name: &str,
    ) -> PyResult<RustArray> {
        self.0
            .combine_with_array(other, op, op_name)
            .map(RustArray::from)
            .map_err(py_err)
    }
}

#[pymethods]
impl RustArray {
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<RustArray> {
        if let Ok(other_ref) = other.extract::<PyRef<RustArray>>() {
            self.combine_with_array(&other_ref, |a, b| a + b, "add")
        } else {
            let scalar = extract_scalar(other)?;
            Ok(RustArray(Array {
                data: scalar_elementwise(&self.data, scalar, |a, b| a + b),
                shape: self.shape,
            }))
        }
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<RustArray> {
        if let Ok(other_ref) = other.extract::<PyRef<RustArray>>() {
            self.combine_with_array(&other_ref, |a, b| a - b, "subtract")
        } else {
            let scalar = extract_scalar(other)?;
            Ok(RustArray(Array {
                data: scalar_elementwise(&self.data, scalar, |a, b| a - b),
                shape: self.shape,
            }))
        }
    }

    /// `1.0 - self.a`-style reflected subtraction (a plain Python float on the left) -
    /// `compute_output_delta`'s own `(a - reference) * a * (1 - a)` formula needs this, not just
    /// array-minus-array.
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<RustArray> {
        let scalar = extract_scalar(other)?;
        Ok(RustArray(Array {
            data: scalar_elementwise(&self.data, scalar, |a, b| b - a),
            shape: self.shape,
        }))
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<RustArray> {
        if let Ok(other_ref) = other.extract::<PyRef<RustArray>>() {
            self.combine_with_array(&other_ref, |a, b| a * b, "multiply")
        } else {
            let scalar = extract_scalar(other)?;
            Ok(RustArray(Array {
                data: scalar_elementwise(&self.data, scalar, |a, b| a * b),
                shape: self.shape,
            }))
        }
    }

    /// `learning_rate * self._grad_W`-style reflected multiplication (a plain Python float on
    /// the left) - multiplication commutes, so this is exactly `__mul__` with the operands
    /// already in the right order for it.
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<RustArray> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<RustArray> {
        if let Ok(other_ref) = other.extract::<PyRef<RustArray>>() {
            self.combine_with_array(&other_ref, |a, b| a / b, "divide")
        } else {
            let scalar = extract_scalar(other)?;
            Ok(RustArray(Array {
                data: scalar_elementwise(&self.data, scalar, |a, b| a / b),
                shape: self.shape,
            }))
        }
    }

    /// A true in-place op (mutates `self.data` directly, `&mut self` with a `()` return) -
    /// `self._grad_W += ...` every training step should not allocate a fresh Python object on
    /// every call.
    fn __iadd__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = self.__add__(other)?;
        self.0 = result.0;
        Ok(())
    }

    /// Not required for correctness (Python falls back to `self = self.__sub__(other)` when
    /// `__isub__` is absent), implemented anyway to avoid an unnecessary allocation on every
    /// `apply_accumulated_gradient` call (`self.W -= ...`).
    fn __isub__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = self.__sub__(other)?;
        self.0 = result.0;
        Ok(())
    }
}
