use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySlice;

/// This core only ever needs a 1D vector or a 2D matrix; general N-dimensional machinery is
/// deliberately not built here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Vector(usize),
    Matrix(usize, usize),
}

impl Shape {
    pub(crate) fn size(&self) -> usize {
        match *self {
            Shape::Vector(n) => n,
            Shape::Matrix(rows, cols) => rows * cols,
        }
    }
}

pub(crate) fn parse_shape(shape: &PyAny) -> PyResult<Shape> {
    if let Ok((rows, cols)) = shape.extract::<(usize, usize)>() {
        Ok(Shape::Matrix(rows, cols))
    } else if let Ok(n) = shape.extract::<usize>() {
        Ok(Shape::Vector(n))
    } else {
        Err(PyTypeError::new_err(
            "shape must be an int (1D) or a (rows, cols) tuple (2D)",
        ))
    }
}

/// One layer's weights/activations/gradients as a flat, row-major f64 buffer plus a shape tag -
/// the Rust-side counterpart to a real numpy `ndarray`, restricted to the subset of numpy's
/// interface this codebase actually uses. Named `Array`, not `PyArray`, to avoid colliding with
/// real numpy's own type of that name.
#[pyclass(name = "Array")]
#[derive(Clone)]
pub struct RustArray {
    pub data: Vec<f64>,
    pub shape: Shape,
}

impl RustArray {
    pub fn from_vector(data: Vec<f64>) -> Self {
        let shape = Shape::Vector(data.len());
        RustArray { data, shape }
    }

    pub fn from_matrix(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        RustArray {
            data,
            shape: Shape::Matrix(rows, cols),
        }
    }
}

#[pymethods]
impl RustArray {
    /// Mirrors `np.array(data)`: a flat Python list of floats builds a 1D array, a nested list
    /// of same-length lists builds a 2D array - the two shapes this whole core ever needs, no
    /// more.
    #[new]
    fn new(data: &PyAny) -> PyResult<Self> {
        if let Ok(rows) = data.extract::<Vec<Vec<f64>>>() {
            let n_rows = rows.len();
            if n_rows == 0 {
                return Err(PyValueError::new_err(
                    "cannot construct a 2D array from zero rows",
                ));
            }
            let n_cols = rows[0].len();
            let mut flat = Vec::with_capacity(n_rows * n_cols);
            for row in &rows {
                if row.len() != n_cols {
                    return Err(PyValueError::new_err(
                        "every row must have the same length",
                    ));
                }
                flat.extend_from_slice(row);
            }
            Ok(RustArray::from_matrix(flat, n_rows, n_cols))
        } else if let Ok(values) = data.extract::<Vec<f64>>() {
            Ok(RustArray::from_vector(values))
        } else {
            Err(PyTypeError::new_err(
                "Array() expects a flat list of numbers (1D) or a nested list of same-length lists (2D)",
            ))
        }
    }

    #[staticmethod]
    fn zeros(shape: &PyAny) -> PyResult<Self> {
        let shape = parse_shape(shape)?;
        Ok(RustArray {
            data: vec![0.0; shape.size()],
            shape,
        })
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> PyObject {
        match self.shape {
            Shape::Vector(n) => (n,).into_py(py),
            Shape::Matrix(rows, cols) => (rows, cols).into_py(py),
        }
    }

    /// `arr[i]` / `arr[i, j]` for single-element reads (two index shapes through the same slot,
    /// matching how Python itself dispatches `arr[i]` vs. `arr[i, j]`), or `arr[:, :-1]` for a
    /// contiguous 2D slice - the one slicing shape this core needs
    /// (`load_mnist_dataset_as_array`'s pixel-vs-label split), not general Python slice
    /// semantics: step must be 1, and there is no fancy/boolean indexing.
    fn __getitem__(&self, py: Python<'_>, index: &PyAny) -> PyResult<PyObject> {
        if let Shape::Matrix(rows, cols) = self.shape {
            if let Ok((row_slice, col_slice)) = index.extract::<(&PySlice, &PySlice)>() {
                let (r0, r1) = Self::resolve_contiguous_range(row_slice, rows)?;
                let (c0, c1) = Self::resolve_contiguous_range(col_slice, cols)?;
                let new_rows = r1.saturating_sub(r0);
                let new_cols = c1.saturating_sub(c0);
                let mut out = Vec::with_capacity(new_rows * new_cols);
                for row in r0..r1 {
                    out.extend_from_slice(&self.data[row * cols + c0..row * cols + c1]);
                }
                return Ok(RustArray::from_matrix(out, new_rows, new_cols).into_py(py));
            }
        }
        let flat_index = self.resolve_index(index)?;
        Ok(self.data[flat_index].into_py(py))
    }

    fn __setitem__(&mut self, index: &PyAny, value: f64) -> PyResult<()> {
        let flat_index = self.resolve_index(index)?;
        self.data[flat_index] = value;
        Ok(())
    }

    fn copy(&self) -> Self {
        RustArray {
            data: self.data.clone(),
            shape: self.shape,
        }
    }

    /// The inverse of `Array(nested_list)`/`Array(flat_list)` (see `new` above) - a flat Python
    /// list for a 1D array, a nested list of same-length lists for a 2D array. `save()`/`load()`
    /// round-trip weights through exactly this pair for JSON serialization.
    fn tolist(&self, py: Python<'_>) -> PyObject {
        match self.shape {
            Shape::Vector(_) => self.data.clone().into_py(py),
            Shape::Matrix(rows, cols) => {
                let nested: Vec<Vec<f64>> = (0..rows)
                    .map(|row| self.data[row * cols..(row + 1) * cols].to_vec())
                    .collect();
                nested.into_py(py)
            }
        }
    }

    /// A no-op on a 1D array (numpy's own `.T` is a no-op there too), a real transpose on 2D -
    /// `#[getter(T)]` keeps the Rust fn name lowercase/snake_case while exposing it to Python as
    /// `.T`, matching `arr.T`'s usage in `ArrayLayer` (`X @ self.W.T`).
    #[getter(T)]
    pub(crate) fn transpose(&self) -> Self {
        match self.shape {
            Shape::Vector(_) => self.clone(),
            Shape::Matrix(rows, cols) => {
                let mut out = vec![0.0; rows * cols];
                for row in 0..rows {
                    for col in 0..cols {
                        out[col * rows + row] = self.data[row * cols + col];
                    }
                }
                RustArray::from_matrix(out, cols, rows)
            }
        }
    }

    /// Reinterprets shape without changing data or order, matching `.reshape()`'s own contract.
    /// Returns an independent array (a full copy of the data),
    /// not a numpy-style view sharing the original buffer - nothing in this core's required
    /// operation set relies on view-aliasing semantics (`load_mnist_dataset_as_array`'s own
    /// reshape-then-slice-then-astype chain already copies at the `.astype` step), so the
    /// simpler, correctness-preserving choice is made here rather than building view machinery
    /// nothing needs yet.
    fn reshape(&self, shape: &PyAny) -> PyResult<Self> {
        let new_shape = parse_shape(shape)?;
        if new_shape.size() != self.data.len() {
            return Err(PyValueError::new_err(format!(
                "cannot reshape array of size {} into shape of size {}",
                self.data.len(),
                new_shape.size()
            )));
        }
        Ok(RustArray {
            data: self.data.clone(),
            shape: new_shape,
        })
    }
}

impl RustArray {
    fn resolve_index(&self, index: &PyAny) -> PyResult<usize> {
        match self.shape {
            Shape::Vector(n) => {
                let i: usize = index.extract().map_err(|_| {
                    PyTypeError::new_err("index into a 1D array must be an int")
                })?;
                if i >= n {
                    return Err(PyIndexError::new_err("index out of range"));
                }
                Ok(i)
            }
            Shape::Matrix(rows, cols) => {
                let (row, col): (usize, usize) = index.extract().map_err(|_| {
                    PyTypeError::new_err(
                        "index into a 2D array must be a (row, col) tuple of ints, or a (slice, slice) tuple",
                    )
                })?;
                if row >= rows || col >= cols {
                    return Err(PyIndexError::new_err("index out of range"));
                }
                Ok(row * cols + col)
            }
        }
    }

    /// Resolves a Python slice against an axis of the given length the same way numpy's own
    /// slicing does (negative indices, an omitted stop, etc.), via `PySlice::indices` - but
    /// rejects any step other than 1, since only contiguous slices are in scope (no fancy or
    /// boolean indexing).
    fn resolve_contiguous_range(slice: &PySlice, len: usize) -> PyResult<(usize, usize)> {
        let indices = slice.indices(len as std::os::raw::c_long)?;
        if indices.step != 1 {
            return Err(PyValueError::new_err(
                "only contiguous (step=1) slices are supported",
            ));
        }
        let start = indices.start.max(0) as usize;
        let stop = indices.stop.max(indices.start) as usize;
        Ok((start, stop))
    }
}
