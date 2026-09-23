use std::sync::OnceLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};

/// `std::thread::available_parallelism()` itself measured at ~50us/call (not cached by the
/// standard library) - queried once, lazily, and cached for the process's lifetime, since the
/// machine's core count doesn't change at runtime.
fn available_parallelism_cached() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

/// The three matmul shape combinations `ArrayLayer`'s own formulas actually use - matrix @
/// vector (`self.W @ x`), vector @ matrix (the interface subset's own "1D x 2D" case, not
/// exercised by the current class design but part of its documented contract), and matrix @
/// matrix (`X @ self.W.T`, `next_layer.delta_batch @ next_layer.W`,
/// `self.delta_batch.T @ input_activation_batch`). All three cases are SIMD-accelerated. The
/// matrix@vector case matters most: it's this codebase's actual `batch_size=1` production path
/// (`fused.rs::layer_forward`/`layer_hidden_delta` call it on every `learn()` step), accounting
/// for ~97% of a fused forward call's cost at the real `dimension=784, hidden=16` shape - without
/// SIMD acceleration it runs ~3.5x slower than numpy at that shape.
pub(crate) fn matmul(a: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    match (a.shape, b.shape) {
        (Shape::Matrix(rows, cols), Shape::Vector(n)) => {
            if cols != n {
                return Err(shape_error(a.shape, b.shape));
            }
            let mut out = vec![0.0; rows];
            for row in 0..rows {
                out[row] = dot_product(&a.data[row * cols..(row + 1) * cols], &b.data);
            }
            Ok(RustArray::from_vector(out))
        }
        (Shape::Vector(n), Shape::Matrix(rows, cols)) => {
            if n != rows {
                return Err(shape_error(a.shape, b.shape));
            }
            // out[0..cols] = sum_k a[k] * b[k*cols .. (k+1)*cols] - structurally identical to
            // matmul_2d_row_range's own row-scaling accumulate (one output "row" instead of
            // many), so it reuses axpy_row directly rather than a bespoke reduction: b's stride
            // over k stays row-contiguous, and there's no per-output-element reduction to worry
            // about staying bit-identical across scalar/SIMD (unlike the matrix@vector case
            // above) - out accumulates sequentially over k regardless of which axpy_row path runs.
            let mut out = vec![0.0; cols];
            for k in 0..rows {
                axpy_row(&mut out, a.data[k], &b.data[k * cols..(k + 1) * cols]);
            }
            Ok(RustArray::from_vector(out))
        }
        (Shape::Matrix(r1, c1), Shape::Matrix(r2, c2)) => {
            if c1 != r2 {
                return Err(shape_error(a.shape, b.shape));
            }
            let mut out = vec![0.0; r1 * c2];
            matmul_2d(&a.data, &b.data, &mut out, r1, c1, c2);
            Ok(RustArray::from_matrix(out, r1, c2))
        }
        (a_shape, b_shape) => Err(shape_error(a_shape, b_shape)),
    }
}

/// `sum(a[i] * b[i] for i in 0..a.len())` - the matrix@vector case's per-row reduction. Unlike
/// `axpy_row` (which vectorizes across the *output* dimension while keeping the reduction over
/// `k` strictly sequential, so its result is bit-identical regardless of which path runs), a
/// dot product's reduction dimension *is* the vectorized dimension - there is no way to sum 4
/// lanes in parallel and then combine them into a single scalar that's also bit-identical to a
/// naive left-to-right sequential sum (float64 addition isn't associative; a different grouping
/// is a different value, typically by 1 ULP or so). So this picks one canonical grouping - 4
/// interleaved partial sums (lane `j` accumulates indices `j, j+4, j+8, ...`), combined pairwise
/// at the end - and uses that *same* grouping in both the scalar fallback and the AVX2 path,
/// which is what actually matters: a training run's result must not depend on which machine
/// happens to run it. Verified bit-identical between the two paths via exact IEEE-754
/// bit-pattern comparison, not just `pytest.approx`. This grouping differs in value from a naive
/// left-to-right sequential sum by last-few-ULPs noise - the same category of divergence as
/// numpy's own internal reduction order already not matching Python's sequential sum, not a new
/// risk category, and every parity check against numpy/the pure-Python reference already
/// tolerates it via rtol, not exact equality.
#[inline]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2_fma(a, b) };
        }
    }
    dot_product_scalar(a, b)
}

/// `(lanes[0] + lanes[1]) + (lanes[2] + lanes[3])` - one fixed combine order, factored out so the
/// scalar and AVX2 paths below can't accidentally diverge by combining their four partial sums
/// differently.
#[inline]
fn combine_lanes(lanes: [f64; 4]) -> f64 {
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// Accumulates `a[start..]`/`b[start..]` sequentially into `initial` - the remainder tail shared
/// by both dot-product paths below once neither has any full 4-wide group left.
#[inline]
fn dot_product_tail(a: &[f64], b: &[f64], start: usize, initial: f64) -> f64 {
    let mut sum = initial;
    for i in start..a.len() {
        sum = a[i].mul_add(b[i], sum);
    }
    sum
}

#[inline]
fn dot_product_scalar(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut lanes = [0.0f64; 4];
    let mut i = 0;
    while i + 4 <= len {
        for lane in 0..4 {
            lanes[lane] = a[i + lane].mul_add(b[i + lane], lanes[lane]);
        }
        i += 4;
    }
    dot_product_tail(a, b, i, combine_lanes(lanes))
}

/// AVX2+FMA path: 4 `f64` lanes per instruction, one `_mm256_fmadd_pd` per 4-element group -
/// lane `j`'s running sum is exactly `dot_product_scalar`'s `lanes[j]`, since a per-lane FMA and
/// `f64::mul_add` compute the same IEEE-754 fused multiply-add. Safety: only ever called after
/// `dot_product`'s runtime `is_x86_feature_detected!` check, same discipline as
/// `axpy_row_avx2_fma` above.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_fma(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    let len = a.len();
    let mut acc_vec = _mm256_setzero_pd();
    let mut i = 0;
    while i + 4 <= len {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        acc_vec = _mm256_fmadd_pd(a_vec, b_vec, acc_vec);
        i += 4;
    }
    let mut lanes = [0.0f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc_vec);
    dot_product_tail(a, b, i, combine_lanes(lanes))
}

/// Threaded row-splitting on top of the size-gated blocking below. Splits the output's row range across
/// `std::thread::scope` workers - safe without `'static` data (each worker borrows `a_data`/
/// `b_data` read-only and writes into its own disjoint slice of `out`, via `split_at_mut`) - only
/// once there's enough total work to plausibly amortize thread spawn overhead.
/// `THREADING_THRESHOLD_FLOPS` is a coarse, deliberately conservative floor (a fraction of a
/// millisecond's worth of naive-loop work), not a tuned constant - measured directly against this
/// codebase's own matmul shapes before being trusted. Splitting by
/// row (not by `k` or `col`) needs no cross-thread reduction: each worker owns complete output
/// rows end to end, so results are bit-identical to the single-threaded path regardless of thread
/// count or scheduling - summation order per output row is unaffected by which thread computes it.
///
/// The flops check runs *before* anything else, including reading `available_parallelism_cached()`
/// - measured directly, `std::thread::available_parallelism()` itself costs ~50us per call (not
/// cached by the standard library, presumably a cgroup/proc filesystem read), which would have
/// silently dominated every one of this codebase's actual small per-call matmuls (already
/// measured in the tens of microseconds) if queried unconditionally on every dispatch. Cached
/// once behind a `OnceLock` and read only when there's already enough work to justify the
/// question.
fn matmul_2d(a_data: &[f64], b_data: &[f64], out: &mut [f64], r1: usize, c1: usize, c2: usize) {
    const THREADING_THRESHOLD_FLOPS: usize = 4_000_000;
    const MAX_THREADS: usize = 8;
    // No rows-per-thread floor is applied here (e.g. requiring >=32 rows/thread): although an
    // isolated microbenchmark of one small-`r1` shape (`(10,512)@(512,784)`) shows unrestricted
    // threading as a slight regression there, measured against the actual target metric (a full
    // mini-batch training step, not one matmul in isolation), adding such a floor makes the real
    // number *worse* (batch_size=512's ratio goes from a 0.98x-1.25x range to 1.24x-1.79x),
    // reproducibly across multiple runs - the isolated shape's regression does not generalize to
    // the composite workload it's actually part of.

    let total_flops = r1 * c1 * c2;
    if total_flops < THREADING_THRESHOLD_FLOPS {
        matmul_2d_row_range(a_data, b_data, out, 0, r1, c1, c2);
        return;
    }

    let thread_count = available_parallelism_cached().min(MAX_THREADS).min(r1);
    if thread_count <= 1 {
        matmul_2d_row_range(a_data, b_data, out, 0, r1, c1, c2);
        return;
    }

    let rows_per_thread = r1.div_ceil(thread_count);
    std::thread::scope(|scope| {
        let mut remaining_out = out;
        let mut row_start = 0;
        while row_start < r1 {
            let row_end = (row_start + rows_per_thread).min(r1);
            let (chunk, rest) = remaining_out.split_at_mut((row_end - row_start) * c2);
            remaining_out = rest;
            scope.spawn(move || {
                matmul_2d_row_range(a_data, b_data, chunk, row_start, row_end, c1, c2);
            });
            row_start = row_end;
        }
    });
}

/// Computes output rows `[row_start, row_end)` into `out_chunk` (row `row_start` maps to
/// `out_chunk[0..c2]`) - the single-threaded and per-thread code path share this, so blocking's
/// own size-gating logic (below) is written once, not duplicated between them.
///
/// `row -> k -> col`, not `row -> col -> k`: accumulates into a whole output row at a
/// time, reading both `a` and `b` row-contiguously.
///
/// Blocked over `row` and `k` when `b` is big enough for it to matter: without blocking, every
/// output row re-streams the *entire* `b` matrix once (`k` ranges over all of `c1`), so if `b`
/// doesn't fit in cache, `b` gets re-fetched from memory once per output row. Blocking caps how
/// much of `b` needs to stay resident at once (one `K_BLOCK`-row slab) and reuses it across
/// `ROW_BLOCK` output rows before moving on. Measured, not assumed: blocking unconditionally was
/// a *regression* at this codebase's actual small layer sizes (e.g. `dimension=784, hidden=16` -
/// `b` is only ~100KB, already cache-resident, so the extra block-boundary bookkeeping was pure
/// overhead - 14% slower), but a genuine 1.3x-2.1x win once `b` exceeds a few hundred KB.
/// `BLOCKING_THRESHOLD_BYTES` is set comfortably below a typical machine's L2 cache size, so
/// blocking only engages once there's real cache pressure for it to relieve. Either path produces
/// the same summation order per output row (k_block sweeps 0..c1 in increasing order, and k
/// sweeps increasing within each block, same as the unblocked loop), so results are
/// bit-identical, not just float64-close, regardless of which path runs.
fn matmul_2d_row_range(
    a_data: &[f64],
    b_data: &[f64],
    out_chunk: &mut [f64],
    row_start: usize,
    row_end: usize,
    c1: usize,
    c2: usize,
) {
    const ROW_BLOCK: usize = 64;
    const K_BLOCK: usize = 64;
    const BLOCKING_THRESHOLD_BYTES: usize = 256 * 1024;
    let b_size_bytes = c1 * c2 * std::mem::size_of::<f64>();

    if b_size_bytes <= BLOCKING_THRESHOLD_BYTES {
        for row in row_start..row_end {
            let out_row = &mut out_chunk[(row - row_start) * c2..(row - row_start + 1) * c2];
            for k in 0..c1 {
                let a_value = a_data[row * c1 + k];
                let b_row = &b_data[k * c2..(k + 1) * c2];
                axpy_row(out_row, a_value, b_row);
            }
        }
        return;
    }

    let mut row_block_start = row_start;
    while row_block_start < row_end {
        let row_block_end = (row_block_start + ROW_BLOCK).min(row_end);
        let mut k_block_start = 0;
        while k_block_start < c1 {
            let k_block_end = (k_block_start + K_BLOCK).min(c1);
            for row in row_block_start..row_block_end {
                let out_row = &mut out_chunk[(row - row_start) * c2..(row - row_start + 1) * c2];
                for k in k_block_start..k_block_end {
                    let a_value = a_data[row * c1 + k];
                    let b_row = &b_data[k * c2..(k + 1) * c2];
                    axpy_row(out_row, a_value, b_row);
                }
            }
            k_block_start = k_block_end;
        }
        row_block_start = row_block_end;
    }
}

/// `out_row[i] = a_value * b_row[i] + out_row[i]` for every `i`, the single accumulate step
/// shared by the blocked and unblocked loops above (and, transitively, by every thread). Fused
/// multiply-add (one rounding, not two) on *every* path - scalar fallback and AVX2 alike - via
/// `f64::mul_add`/`_mm256_fmadd_pd`, so this call produces the same bits whether or not the
/// running machine has AVX2, matching the bit-identical invariant this file's blocking/threading
/// hold to. A plain `_mm256_mul_pd` + `_mm256_add_pd` pair would vectorize fine but round twice
/// per element instead of once, silently reintroducing a bit-level divergence between this path
/// and any non-AVX2 fallback - FMA is the only way to keep both the speed and the invariant.
#[inline]
fn axpy_row(out_row: &mut [f64], a_value: f64, b_row: &[f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { axpy_row_avx2_fma(out_row, a_value, b_row) };
            return;
        }
    }
    axpy_row_scalar(out_row, a_value, b_row);
}

#[inline]
fn axpy_row_scalar(out_row: &mut [f64], a_value: f64, b_row: &[f64]) {
    for col in 0..out_row.len() {
        out_row[col] = a_value.mul_add(b_row[col], out_row[col]);
    }
}

/// AVX2+FMA path: 4 `f64` lanes per instruction. Safety: only ever called after
/// `axpy_row`'s runtime `is_x86_feature_detected!` check confirms both features are present -
/// `#[target_feature]` functions are unsafe to call directly because the compiler can't itself
/// prove that precondition. Uses unaligned loads/stores (`out_row`/`b_row` are arbitrary slices
/// into a larger buffer, not independently aligned) and a scalar `mul_add` tail for the
/// `c2 % 4` remainder, so the result matches `axpy_row_scalar` bit for bit regardless of `c2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_row_avx2_fma(out_row: &mut [f64], a_value: f64, b_row: &[f64]) {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};

    let len = out_row.len();
    let a_vec = _mm256_set1_pd(a_value);
    let mut col = 0;
    while col + 4 <= len {
        let b_vec = _mm256_loadu_pd(b_row.as_ptr().add(col));
        let acc_vec = _mm256_loadu_pd(out_row.as_ptr().add(col));
        let result = _mm256_fmadd_pd(a_vec, b_vec, acc_vec);
        _mm256_storeu_pd(out_row.as_mut_ptr().add(col), result);
        col += 4;
    }
    while col < len {
        out_row[col] = a_value.mul_add(b_row[col], out_row[col]);
        col += 1;
    }
}

fn shape_error(a_shape: Shape, b_shape: Shape) -> PyErr {
    PyValueError::new_err(format!(
        "cannot matmul arrays of shape {:?} and {:?}",
        a_shape, b_shape
    ))
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
/// `np.outer(delta, input_layer.a)`. Exposed as a free function (`indrajala_ml_array.outer(a, b)`),
/// matching `np.outer`'s own call style rather than an operator.
#[pyfunction]
pub fn outer(a: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    match (a.shape, b.shape) {
        (Shape::Vector(m), Shape::Vector(n)) => {
            let mut out = Vec::with_capacity(m * n);
            for &a_value in &a.data {
                for &b_value in &b.data {
                    out.push(a_value * b_value);
                }
            }
            Ok(RustArray::from_matrix(out, m, n))
        }
        (a_shape, b_shape) => Err(PyValueError::new_err(format!(
            "outer requires two 1D vectors, got shapes {:?} and {:?}",
            a_shape, b_shape
        ))),
    }
}
