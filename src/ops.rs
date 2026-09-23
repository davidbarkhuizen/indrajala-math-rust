use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};

/// Accepts a Python int or float as a scalar operand (`grad_W / batch_size` passes a Python
/// int) - numpy itself accepts either for a scalar operation, so this core does too.
fn extract_scalar(value: &PyAny) -> PyResult<f64> {
    if let Ok(v) = value.extract::<f64>() {
        return Ok(v);
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok(v as f64);
    }
    Err(PyTypeError::new_err("expected a numeric scalar"))
}

// pub(crate): fused.rs's layer_apply_accumulated_gradient reuses this directly rather than
// hand-writing the same zip/map loop for its own W/grad_W and b/grad_b scaled-subtract pairs.
pub(crate) fn same_shape_elementwise(a: &[f64], b: &[f64], op: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| op(x, y)).collect()
}

fn scalar_elementwise(a: &[f64], scalar: f64, op: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    a.iter().map(|&x| op(x, scalar)).collect()
}

/// Broadcasts `vector` (length `cols`) across every row of a `rows x cols` matrix - the one
/// broadcasting case this core needs, alongside plain same-shape elementwise addition
/// (`X @ self.W.T + self.b`).
fn broadcast_row(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    vector: &[f64],
    op: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            out.push(op(matrix[row * cols + col], vector[col]));
        }
    }
    out
}

impl RustArray {
    pub(crate) fn combine_with_array(
        &self,
        other: &RustArray,
        op: impl Fn(f64, f64) -> f64,
        op_name: &str,
    ) -> PyResult<RustArray> {
        match (self.shape, other.shape) {
            (Shape::Matrix(rows, cols), Shape::Vector(n)) if n == cols => Ok(RustArray::from_matrix(
                broadcast_row(&self.data, rows, cols, &other.data, &op),
                rows,
                cols,
            )),
            (Shape::Vector(n), Shape::Matrix(rows, cols)) if n == cols => Ok(RustArray::from_matrix(
                broadcast_row(&other.data, rows, cols, &self.data, |matrix_value, vector_value| {
                    op(vector_value, matrix_value)
                }),
                rows,
                cols,
            )),
            (a_shape, b_shape) if a_shape == b_shape => Ok(RustArray {
                data: same_shape_elementwise(&self.data, &other.data, op),
                shape: self.shape,
            }),
            (a_shape, b_shape) => Err(PyValueError::new_err(format!(
                "cannot {op_name} arrays of shape {:?} and {:?}",
                a_shape, b_shape
            ))),
        }
    }
}

#[pymethods]
impl RustArray {
    fn __add__(&self, other: &PyAny) -> PyResult<RustArray> {
        if let Ok(other_ref) = other.extract::<PyRef<RustArray>>() {
            self.combine_with_array(&other_ref, |a, b| a + b, "add")
        } else {
            let scalar = extract_scalar(other)?;
            Ok(RustArray {
                data: scalar_elementwise(&self.data, scalar, |a, b| a + b),
                shape: self.shape,
            })
        }
    }

    fn __sub__(&self, other: &PyAny) -> PyResult<RustArray> {
        if let Ok(other_ref) = other.extract::<PyRef<RustArray>>() {
            self.combine_with_array(&other_ref, |a, b| a - b, "subtract")
        } else {
            let scalar = extract_scalar(other)?;
            Ok(RustArray {
                data: scalar_elementwise(&self.data, scalar, |a, b| a - b),
                shape: self.shape,
            })
        }
    }

    /// `1.0 - self.a`-style reflected subtraction (a plain Python float on the left) -
    /// `compute_output_delta`'s own `(a - reference) * a * (1 - a)` formula needs this, not just
    /// array-minus-array.
    fn __rsub__(&self, other: &PyAny) -> PyResult<RustArray> {
        let scalar = extract_scalar(other)?;
        Ok(RustArray {
            data: scalar_elementwise(&self.data, scalar, |a, b| b - a),
            shape: self.shape,
        })
    }

    fn __mul__(&self, other: &PyAny) -> PyResult<RustArray> {
        if let Ok(other_ref) = other.extract::<PyRef<RustArray>>() {
            self.combine_with_array(&other_ref, |a, b| a * b, "multiply")
        } else {
            let scalar = extract_scalar(other)?;
            Ok(RustArray {
                data: scalar_elementwise(&self.data, scalar, |a, b| a * b),
                shape: self.shape,
            })
        }
    }

    /// `learning_rate * self._grad_W`-style reflected multiplication (a plain Python float on
    /// the left) - multiplication commutes, so this is exactly `__mul__` with the operands
    /// already in the right order for it.
    fn __rmul__(&self, other: &PyAny) -> PyResult<RustArray> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: &PyAny) -> PyResult<RustArray> {
        if let Ok(other_ref) = other.extract::<PyRef<RustArray>>() {
            self.combine_with_array(&other_ref, |a, b| a / b, "divide")
        } else {
            let scalar = extract_scalar(other)?;
            Ok(RustArray {
                data: scalar_elementwise(&self.data, scalar, |a, b| a / b),
                shape: self.shape,
            })
        }
    }

    /// A true in-place op (mutates `self.data` directly, `&mut self` with a `()` return) -
    /// `self._grad_W += ...` every training step should not allocate a fresh Python object on
    /// every call.
    fn __iadd__(&mut self, other: &PyAny) -> PyResult<()> {
        let result = self.__add__(other)?;
        self.data = result.data;
        self.shape = result.shape;
        Ok(())
    }

    /// Not required for correctness (Python falls back to `self = self.__sub__(other)` when
    /// `__isub__` is absent), implemented anyway to avoid an unnecessary allocation on every
    /// `apply_accumulated_gradient` call (`self.W -= ...`).
    fn __isub__(&mut self, other: &PyAny) -> PyResult<()> {
        let result = self.__sub__(other)?;
        self.data = result.data;
        self.shape = result.shape;
        Ok(())
    }
}
