use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PyTuple};

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
        Ok(RustArray::from_vector(self.data[i * cols..(i + 1) * cols].to_vec()))
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
            out.extend_from_slice(&self.data[i * cols..(i + 1) * cols]);
        }
        Ok(RustArray::from_matrix(out, indices.len(), cols))
    }

    #[staticmethod]
    fn zeros(shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape = parse_shape(shape)?;
        Ok(RustArray {
            data: vec![0.0; shape.size()],
            shape,
        })
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        match self.shape {
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
        if let Shape::Matrix(rows, cols) = self.shape {
            if let Ok((row_slice, col_slice)) = index.extract::<(Bound<'_, PySlice>, Bound<'_, PySlice>)>() {
                let (r0, r1) = Self::resolve_contiguous_range(&row_slice, rows)?;
                let (c0, c1) = Self::resolve_contiguous_range(&col_slice, cols)?;
                let new_rows = r1.saturating_sub(r0);
                let new_cols = c1.saturating_sub(c0);
                let mut out = Vec::with_capacity(new_rows * new_cols);
                for row in r0..r1 {
                    out.extend_from_slice(&self.data[row * cols + c0..row * cols + c1]);
                }
                return Ok(RustArray::from_matrix(out, new_rows, new_cols)
                    .into_pyobject(py)?
                    .into_any());
            }
        }
        let flat_index = self.resolve_index(index)?;
        Ok(self.data[flat_index].into_pyobject(py)?.into_any())
    }

    fn __setitem__(&mut self, index: &Bound<'_, PyAny>, value: f64) -> PyResult<()> {
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
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.shape {
            Shape::Vector(_) => self.data.clone().into_pyobject(py),
            Shape::Matrix(rows, cols) => {
                let nested: Vec<Vec<f64>> = (0..rows)
                    .map(|row| self.data[row * cols..(row + 1) * cols].to_vec())
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
        match self.shape {
            Shape::Vector(_) => self.clone(),
            Shape::Matrix(rows, cols) => {
                // In 8 x 8 blocks, so each block reads 8 source lines and writes 8 whole output
                // lines. A row-by-row loop writes a line per element at a stride of `rows`; at
                // 512 rows that stride is 4 KB, every write lands in the same L1 set, and the
                // (512, 30) delta_batch.T of the dense accumulate took 87-107 µs against numpy's
                // 8-12. A copy, so bit-identical in any order.
                const BLOCK: usize = 8;
                let mut out = vec![0.0; rows * cols];
                #[cfg(target_arch = "x86_64")]
                let avx2 = std::is_x86_feature_detected!("avx2");
                for row_start in (0..rows).step_by(BLOCK) {
                    let row_end = (row_start + BLOCK).min(rows);
                    for col_start in (0..cols).step_by(BLOCK) {
                        let col_end = (col_start + BLOCK).min(cols);
                        #[cfg(target_arch = "aarch64")]
                        // Safety: the block lies within both `self.data` and `out`.
                        unsafe {
                            transpose_block_neon(
                                &self.data,
                                &mut out,
                                rows,
                                cols,
                                row_start..row_end,
                                col_start..col_end,
                            );
                        }
                        #[cfg(target_arch = "x86_64")]
                        if avx2 {
                            // Safety: AVX2 detected above; the block lies within both `self.data` and `out`.
                            unsafe {
                                transpose_block_avx2(
                                    &self.data,
                                    &mut out,
                                    rows,
                                    cols,
                                    row_start..row_end,
                                    col_start..col_end,
                                );
                            }
                        } else {
                            transpose_block(&self.data, &mut out, rows, cols, row_start..row_end, col_start..col_end);
                        }
                        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                        transpose_block(&self.data, &mut out, rows, cols, row_start..row_end, col_start..col_end);
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
    fn reshape(&self, shape: &Bound<'_, PyAny>) -> PyResult<Self> {
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
    fn matrix_dims(&self, op: &str) -> PyResult<(usize, usize)> {
        match self.shape {
            Shape::Matrix(rows, cols) => Ok((rows, cols)),
            Shape::Vector(_) => Err(PyValueError::new_err(format!("{op} needs a 2D array"))),
        }
    }

    fn resolve_index(&self, index: &Bound<'_, PyAny>) -> PyResult<usize> {
        match self.shape {
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

/// One block of `transpose`: `out[col * rows + row] = data[row * cols + col]` over the block's
/// rows and columns, column by column. A copy, so bit-identical in any order.
#[cfg_attr(target_arch = "aarch64", expect(dead_code, reason = "NEON replaces it on aarch64"))]
fn transpose_block(
    data: &[f64],
    out: &mut [f64],
    rows: usize,
    cols: usize,
    block_rows: std::ops::Range<usize>,
    block_cols: std::ops::Range<usize>,
) {
    for col in block_cols {
        for row in block_rows.clone() {
            out[col * rows + row] = data[row * cols + col];
        }
    }
}

/// `transpose_block` in AVX2 (the crate kernels workplan's D4): 4 x 4 sub-blocks, four source rows'
/// quads loaded, interleaved in pairs (`_mm256_unpacklo_pd`, `_mm256_unpackhi_pd`) and their 128-bit
/// halves recombined (`_mm256_permute2f128_pd`) into four output rows' quads, so each load and store
/// moves four values; the last rows and columns short of a quad copied one value at a time. A copy,
/// so bit-identical to the scalar block. Safety: AVX2 is available, and the block lies within `data`
/// (`rows x cols`) and `out`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_block_avx2(
    data: &[f64],
    out: &mut [f64],
    rows: usize,
    cols: usize,
    block_rows: std::ops::Range<usize>,
    block_cols: std::ops::Range<usize>,
) {
    use std::arch::x86_64::{
        _mm256_loadu_pd, _mm256_permute2f128_pd, _mm256_storeu_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };

    let (src, dst) = (data.as_ptr(), out.as_mut_ptr());
    let quad_rows = block_rows.start + (block_rows.len() & !3);
    let quad_cols = block_cols.start + (block_cols.len() & !3);
    for row in (block_rows.start..quad_rows).step_by(4) {
        for col in (block_cols.start..quad_cols).step_by(4) {
            let r0 = _mm256_loadu_pd(src.add(row * cols + col));
            let r1 = _mm256_loadu_pd(src.add((row + 1) * cols + col));
            let r2 = _mm256_loadu_pd(src.add((row + 2) * cols + col));
            let r3 = _mm256_loadu_pd(src.add((row + 3) * cols + col));
            // [r0[0], r1[0], r0[2], r1[2]], [r0[1], r1[1], r0[3], r1[3]], and likewise rows 2 and 3
            let (even01, odd01) = (_mm256_unpacklo_pd(r0, r1), _mm256_unpackhi_pd(r0, r1));
            let (even23, odd23) = (_mm256_unpacklo_pd(r2, r3), _mm256_unpackhi_pd(r2, r3));
            // 0x20 takes both low halves (columns 0 and 1), 0x31 both high halves (columns 2 and 3)
            let quads = [
                _mm256_permute2f128_pd(even01, even23, 0x20),
                _mm256_permute2f128_pd(odd01, odd23, 0x20),
                _mm256_permute2f128_pd(even01, even23, 0x31),
                _mm256_permute2f128_pd(odd01, odd23, 0x31),
            ];
            for (offset, quad) in quads.into_iter().enumerate() {
                _mm256_storeu_pd(dst.add((col + offset) * rows + row), quad);
            }
        }
        for col in quad_cols..block_cols.end {
            for r in row..row + 4 {
                *dst.add(col * rows + r) = *src.add(r * cols + col);
            }
        }
    }
    for row in quad_rows..block_rows.end {
        for col in block_cols.clone() {
            *dst.add(col * rows + row) = *src.add(row * cols + col);
        }
    }
}

/// `transpose_block` in NEON (the aarch64 workplan's stage 7): 2 x 2 sub-blocks, two source rows'
/// pairs loaded and zipped (`vzip1q_f64`, `vzip2q_f64`) into two output rows' pairs, so each load and
/// store moves two values; an odd last row or column copied one value at a time. A copy, so
/// bit-identical to the scalar block. Safety: the block lies within `data` (`rows x cols`) and `out`.
#[cfg(target_arch = "aarch64")]
unsafe fn transpose_block_neon(
    data: &[f64],
    out: &mut [f64],
    rows: usize,
    cols: usize,
    block_rows: std::ops::Range<usize>,
    block_cols: std::ops::Range<usize>,
) {
    use std::arch::aarch64::{vld1q_f64, vst1q_f64, vzip1q_f64, vzip2q_f64};

    let (src, dst) = (data.as_ptr(), out.as_mut_ptr());
    let even_rows = block_rows.start + (block_rows.len() & !1);
    let even_cols = block_cols.start + (block_cols.len() & !1);
    for row in (block_rows.start..even_rows).step_by(2) {
        for col in (block_cols.start..even_cols).step_by(2) {
            let upper = vld1q_f64(src.add(row * cols + col));
            let lower = vld1q_f64(src.add((row + 1) * cols + col));
            vst1q_f64(dst.add(col * rows + row), vzip1q_f64(upper, lower));
            vst1q_f64(dst.add((col + 1) * rows + row), vzip2q_f64(upper, lower));
        }
        for col in even_cols..block_cols.end {
            *dst.add(col * rows + row) = *src.add(row * cols + col);
            *dst.add(col * rows + row + 1) = *src.add((row + 1) * cols + col);
        }
    }
    for row in even_rows..block_rows.end {
        for col in block_cols.clone() {
            *dst.add(col * rows + row) = *src.add(row * cols + col);
        }
    }
}
