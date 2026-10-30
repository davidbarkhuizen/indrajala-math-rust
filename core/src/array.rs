//! The array type: a flat, row-major `f64` buffer with a 1D or 2D shape.

/// This core only ever needs a 1D vector or a 2D matrix; general N-dimensional machinery is
/// deliberately not built here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// A 1D vector of this length.
    Vector(usize),
    /// A `rows x cols` matrix, row-major.
    Matrix(usize, usize),
}

impl Shape {
    /// The number of values an array of this shape holds.
    pub fn size(&self) -> usize {
        match *self {
            Shape::Vector(n) => n,
            Shape::Matrix(rows, cols) => rows * cols,
        }
    }
}

/// An array: `data` holds `shape.size()` values, row-major. Every operation borrows its operands
/// (`&Array`) and returns a new array.
#[derive(Clone, Debug, PartialEq)]
pub struct Array {
    /// The values, row-major: `shape.size()` of them.
    pub data: Vec<f64>,
    /// 1D or 2D.
    pub shape: Shape,
}

impl Array {
    /// A 1D array of `data`.
    pub fn from_vector(data: Vec<f64>) -> Self {
        let shape = Shape::Vector(data.len());
        Array { data, shape }
    }

    /// A `rows x cols` array of `data`, row-major.
    pub fn from_matrix(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        Array {
            data,
            shape: Shape::Matrix(rows, cols),
        }
    }
}

impl Array {
    /// This array's transpose: a copy on 1D (numpy's `.T` is a no-op there), the real transpose on 2D.
    #[inline]
    pub fn transpose(&self) -> Array {
        match self.shape {
            Shape::Vector(_) => Array {
                data: self.data.clone(),
                shape: self.shape,
            },
            Shape::Matrix(rows, cols) => {
                // In 8 x 8 blocks, so each block reads 8 source lines and writes 8 whole output
                // lines. A row-by-row loop writes a line per element at a stride of `rows`; at
                // 512 rows that stride is 4 KB, every write lands in the same L1 set, and the
                // (512, 30) delta_batch.T of the dense accumulate took 87-107 µs against numpy's
                // 8-12. A copy, so bit-identical in any order.
                const BLOCK: usize = 8;
                let mut out = vec![0.0; rows * cols];
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
                        #[cfg(not(target_arch = "aarch64"))]
                        transpose_block(&self.data, &mut out, rows, cols, row_start..row_end, col_start..col_end);
                    }
                }
                Array::from_matrix(out, cols, rows)
            }
        }
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
