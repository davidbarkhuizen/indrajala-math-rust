//! Elementwise arithmetic: same-shape, scalar and one broadcast row.

use crate::array::{Array, Shape};
use crate::Error;

// fused's layer_apply_accumulated_gradient reuses this directly rather than hand-writing the same zip/map loop for its own W/grad_W and b/grad_b scaled-subtract pairs.
pub(crate) fn same_shape_elementwise(a: &[f64], b: &[f64], op: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| op(x, y)).collect()
}

/// `op(a[i], scalar)` for every value of `a`.
pub fn scalar_elementwise(a: &[f64], scalar: f64, op: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    a.iter().map(|&x| op(x, scalar)).collect()
}

/// Broadcasts `vector` (length `cols`) across every row of a `rows x cols` matrix - the one
/// broadcasting case this core needs, alongside plain same-shape elementwise addition
/// (`X @ self.W.T + self.b`).
pub(crate) fn broadcast_row(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    vector: &[f64],
    op: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows * cols);
    for row in matrix.chunks_exact(cols.max(1)) {
        out.extend(row.iter().zip(vector).map(|(&m, &v)| op(m, v)));
    }
    out
}

impl Array {
    /// `self op other`: the same shape, or a matrix and a vector of its row length broadcast
    /// across its rows (either side).
    pub fn combine_with_array(
        &self,
        other: &Array,
        op: impl Fn(f64, f64) -> f64,
        op_name: &str,
    ) -> Result<Array, Error> {
        match (self.shape, other.shape) {
            (Shape::Matrix(rows, cols), Shape::Vector(n)) if n == cols => Ok(Array::from_matrix(
                broadcast_row(&self.data, rows, cols, &other.data, &op),
                rows,
                cols,
            )),
            (Shape::Vector(n), Shape::Matrix(rows, cols)) if n == cols => Ok(Array::from_matrix(
                broadcast_row(&other.data, rows, cols, &self.data, |matrix_value, vector_value| {
                    op(vector_value, matrix_value)
                }),
                rows,
                cols,
            )),
            (a_shape, b_shape) if a_shape == b_shape => Ok(Array {
                data: same_shape_elementwise(&self.data, &other.data, op),
                shape: self.shape,
            }),
            (a_shape, b_shape) => Err(Error::invalid(format!(
                "cannot {op_name} arrays of shape {:?} and {:?}",
                a_shape, b_shape
            ))),
        }
    }
}

pub(crate) fn require_same_shape(a: &Array, b: &Array, context: &str) -> Result<(), Error> {
    if a.shape != b.shape {
        return Err(Error::invalid(format!(
            "{context} requires matching shapes, got {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    Ok(())
}
