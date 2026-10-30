use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PyTuple};

use crate::linalg::py_err;
use indrajala_math::Array;
pub use indrajala_math::Shape;

pub(crate) fn parse_shape(shape: &Bound<'_, PyAny>) -> PyResult<Shape> {
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
#[pyclass(name = "Array", skip_from_py_object)]
#[derive(Clone)]
pub struct RustArray(pub Array);

/// The core array's methods (`a.data()`, `a.shape()`) and every core operation (`&Array`) reach it
/// directly, so nothing is copied at the boundary.
impl std::ops::Deref for RustArray {
    type Target = Array;

    fn deref(&self) -> &Array {
        &self.0
    }
}

impl std::ops::DerefMut for RustArray {
    fn deref_mut(&mut self) -> &mut Array {
        &mut self.0
    }
}

impl RustArray {
    pub fn from_vector(data: Vec<f64>) -> Self {
        RustArray(Array::from_vector(data))
    }

    pub fn from_matrix(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        RustArray(Array::from_matrix(data, rows, cols))
    }
}

/// The core's result moved in: no copy.
impl From<Array> for RustArray {
    fn from(array: Array) -> Self {
        RustArray(array)
    }
}

#[pymethods]
impl RustArray {
    /// Mirrors `np.array(data)`: a flat Python list of floats builds a 1D array, a nested list
    /// of same-length lists builds a 2D array - the two shapes this whole core ever needs, no
    /// more.
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(rows) = data.extract::<Vec<Vec<f64>>>() {
            let n_rows = rows.len();
            if n_rows == 0 {
                return Err(PyValueError::new_err("cannot construct a 2D array from zero rows"));
            }
            let n_cols = rows[0].len();
            let mut flat = Vec::with_capacity(n_rows * n_cols);
            for row in &rows {
                if row.len() != n_cols {
                    return Err(PyValueError::new_err("every row must have the same length"));
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

    /// A 2D array from any sequence of equal-length float sequences (tuples as well as lists),
    /// written into one pre-sized buffer, for converting a whole dataset once. Full MNIST
    /// (60000 tuples of 784 floats) takes about 0.4 s, against 1.5 s through `Array(...)`;
    /// nearly all of the difference is reading tuples and lists by index rather than through
    /// Python iteration. Same values and errors as `Array(nested)`.
    #[staticmethod]
    fn from_rows(rows: &Bound<'_, PyAny>) -> PyResult<Self> {
        let n_rows = rows.len()?;
        if n_rows == 0 {
            return Err(PyValueError::new_err("cannot construct a 2D array from zero rows"));
        }
        let n_cols = rows.get_item(0)?.len()?;
        let mut flat = Vec::with_capacity(n_rows * n_cols);
        for row in rows.try_iter()? {
            let row = row?;
            if row.len()? != n_cols {
                return Err(PyValueError::new_err("every row must have the same length"));
            }
            // tuples and lists are read by index, which is much faster than Python iteration
            if let Ok(tuple) = row.cast::<PyTuple>() {
                for value in tuple.as_slice() {
                    flat.push(value.extract::<f64>()?);
                }
            } else if let Ok(list) = row.cast::<PyList>() {
                for value in list.iter() {
                    flat.push(value.extract::<f64>()?);
                }
            } else {
                for value in row.try_iter()? {
                    flat.push(value?.extract::<f64>()?);
                }
            }
        }
        if flat.len() != n_rows * n_cols {
            return Err(PyValueError::new_err("every row must have the same length"));
        }
        Ok(RustArray::from_matrix(flat, n_rows, n_cols))
    }

    /// Row `i` of a 2D array as a new 1D array (a copy: this core has no views).
    fn row(&self, i: usize) -> PyResult<Self> {
        let (rows, cols) = self.matrix_dims("row")?;
        if i >= rows {
            return Err(PyIndexError::new_err("row index out of range"));
        }
        Ok(RustArray::from_vector(self.0.data()[i * cols..(i + 1) * cols].to_vec()))
    }

    /// The given rows of a 2D array, in the given order (repeats allowed), as a new 2D array -
    /// numpy's `arr[indices]` for a list of row indices.
    fn take_rows(&self, indices: Vec<usize>) -> PyResult<Self> {
        let (rows, cols) = self.matrix_dims("take_rows")?;
        if indices.is_empty() {
            return Err(PyValueError::new_err("take_rows needs at least one row index"));
        }
        let mut out = Vec::with_capacity(indices.len() * cols);
        for &i in &indices {
            if i >= rows {
                return Err(PyIndexError::new_err("row index out of range"));
            }
            out.extend_from_slice(&self.0.data()[i * cols..(i + 1) * cols]);
        }
        Ok(RustArray::from_matrix(out, indices.len(), cols))
    }

    #[staticmethod]
    fn zeros(shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape = parse_shape(shape)?;
        Ok(RustArray(Array::new(vec![0.0; shape.size()], shape).map_err(py_err)?))
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        match self.0.shape() {
            Shape::Vector(n) => PyTuple::new(py, [n]),
            Shape::Matrix(rows, cols) => PyTuple::new(py, [rows, cols]),
        }
    }

    /// `arr[i]` / `arr[i, j]` for single-element reads (two index shapes through the same slot,
    /// matching how Python itself dispatches `arr[i]` vs. `arr[i, j]`), or `arr[:, :-1]` for a
    /// contiguous 2D slice - the one slicing shape this core needs
    /// (`load_mnist_dataset_as_array`'s pixel-vs-label split), not general Python slice
    /// semantics: step must be 1, and there is no fancy/boolean indexing.
    fn __getitem__<'py>(&self, py: Python<'py>, index: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if let Shape::Matrix(rows, cols) = self.0.shape() {
            if let Ok((row_slice, col_slice)) = index.extract::<(Bound<'_, PySlice>, Bound<'_, PySlice>)>() {
                let (r0, r1) = Self::resolve_contiguous_range(&row_slice, rows)?;
                let (c0, c1) = Self::resolve_contiguous_range(&col_slice, cols)?;
                let new_rows = r1.saturating_sub(r0);
                let new_cols = c1.saturating_sub(c0);
                let mut out = Vec::with_capacity(new_rows * new_cols);
                for row in r0..r1 {
                    out.extend_from_slice(&self.0.data()[row * cols + c0..row * cols + c1]);
                }
                return Ok(RustArray::from_matrix(out, new_rows, new_cols)
                    .into_pyobject(py)?
                    .into_any());
            }
        }
        let flat_index = self.resolve_index(index)?;
        Ok(self.0.data()[flat_index].into_pyobject(py)?.into_any())
    }

    fn __setitem__(&mut self, index: &Bound<'_, PyAny>, value: f64) -> PyResult<()> {
        let flat_index = self.resolve_index(index)?;
        self.0.data_mut()[flat_index] = value;
        Ok(())
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    /// The inverse of `Array(nested_list)`/`Array(flat_list)` (see `new` above) - a flat Python
    /// list for a 1D array, a nested list of same-length lists for a 2D array. `save()`/`load()`
    /// round-trip weights through exactly this pair for JSON serialization.
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.0.shape() {
            Shape::Vector(_) => self.0.data().to_vec().into_pyobject(py),
            Shape::Matrix(rows, cols) => {
                let nested: Vec<Vec<f64>> = (0..rows)
                    .map(|row| self.0.data()[row * cols..(row + 1) * cols].to_vec())
                    .collect();
                nested.into_pyobject(py)
            }
        }
    }

    /// A no-op on a 1D array (numpy's own `.T` is a no-op there too), a real transpose on 2D -
    /// `#[getter(T)]` keeps the Rust fn name lowercase/snake_case while exposing it to Python as
    /// `.T`, matching `arr.T`'s usage in `ArrayLayer` (`X @ self.W.T`).
    #[getter(T)]
    pub(crate) fn transpose(&self) -> Self {
        match self.0.shape() {
            Shape::Vector(_) => self.clone(),
            Shape::Matrix(..) => self.0.transpose().into(),
        }
    }

    /// Reinterprets shape without changing data or order, matching `.reshape()`'s own contract.
    /// Returns an independent array (a full copy of the data),
    /// not a numpy-style view sharing the original buffer - nothing in this core's required
    /// operation set relies on view-aliasing semantics (`load_mnist_dataset_as_array`'s own
    /// reshape-then-slice-then-astype chain already copies at the `.astype` step), so the
    /// simpler, correctness-preserving choice is made here rather than building view machinery
    /// nothing needs yet.
    fn reshape(&self, shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        let new_shape = parse_shape(shape)?;
        if new_shape.size() != self.0.data().len() {
            return Err(PyValueError::new_err(format!(
                "cannot reshape array of size {} into shape of size {}",
                self.0.data().len(),
                new_shape.size()
            )));
        }
        Ok(RustArray(
            Array::new(self.0.data().to_vec(), new_shape).map_err(py_err)?,
        ))
    }
}

impl RustArray {
    fn matrix_dims(&self, op: &str) -> PyResult<(usize, usize)> {
        match self.0.shape() {
            Shape::Matrix(rows, cols) => Ok((rows, cols)),
            Shape::Vector(_) => Err(PyValueError::new_err(format!("{op} needs a 2D array"))),
        }
    }

    fn resolve_index(&self, index: &Bound<'_, PyAny>) -> PyResult<usize> {
        match self.0.shape() {
            Shape::Vector(n) => {
                let i: usize = index
                    .extract()
                    .map_err(|_| PyTypeError::new_err("index into a 1D array must be an int"))?;
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
    fn resolve_contiguous_range(slice: &Bound<'_, PySlice>, len: usize) -> PyResult<(usize, usize)> {
        let indices = slice.indices(len as isize)?;
        if indices.step != 1 {
            return Err(PyValueError::new_err("only contiguous (step=1) slices are supported"));
        }
        let start = indices.start.max(0) as usize;
        let stop = indices.stop.max(indices.start) as usize;
        Ok((start, stop))
    }
}
