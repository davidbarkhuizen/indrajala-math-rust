use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};

/// `std::thread::available_parallelism()` itself measured at ~50us/call (not cached by the
/// standard library) - queried once, lazily, and cached for the process's lifetime, since the
/// machine's core count doesn't change at runtime.
fn available_parallelism_cached() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

/// The three matmul shape combinations `ArrayLayer`'s own formulas actually use - matrix @
/// vector (`self.W @ x`), vector @ matrix (`self.W.T @ self.delta`, computed as `delta @ W` by
/// `fused.rs::layer_downstream` on every single-example backward step), and matrix @
/// matrix (`next_layer.delta_batch @ next_layer.W`, `self.delta_batch.T @ input_activation_batch`;
/// `X @ self.W.T` is `matmul_nt` below). All three cases are SIMD-accelerated. The
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
            dot_products_into(&a.data, cols, &b.data, &mut out);
            Ok(RustArray::from_vector(out))
        }
        (Shape::Vector(n), Shape::Matrix(rows, cols)) => {
            if n != rows {
                return Err(shape_error(a.shape, b.shape));
            }
            // a one-row matrix @ matrix product: the same kernel and FMA chain per output, so
            // each row of a matrix @ matrix product has these bits. Accumulating the whole output
            // row per k instead (one load/FMA/store pass over it per row of b) took 43-52 us at
            // 32 x 5408, where the tiled kernel's registers took 21-22.
            let mut out = vec![0.0; cols];
            tiled_row_range::<OVERWRITE>(&a.data, &b.data, &mut out, Panel::whole(0, 1, rows, cols, 1));
            Ok(RustArray::from_vector(out))
        }
        (Shape::Matrix(r1, c1), Shape::Matrix(r2, c2)) => {
            if c1 != r2 {
                return Err(shape_error(a.shape, b.shape));
            }
            let mut out = vec![0.0; r1 * c2];
            matmul_2d::<OVERWRITE>(&a.data, &b.data, &mut out, r1, c1, c2);
            Ok(RustArray::from_matrix(out, r1, c2))
        }
        (a_shape, b_shape) => Err(shape_error(a_shape, b_shape)),
    }
}

/// `sum(a[i] * b[i] for i in 0..a.len())` - the matrix@vector case's per-row reduction. Unlike
/// `tiled_row_range` (which vectorizes across the *output* dimension while keeping the reduction
/// over `k` strictly sequential, so its result is bit-identical regardless of which path runs), a
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

/// `out[r] = dot_product(rows[r], v)` for each of the `out.len()` rows of `rows` (row-major, `k`
/// wide), bit for bit. `dot_product`'s one accumulator makes every FMA wait on the one before it
/// (a latency chain of `k / 4` dependent FMAs), so a lone matrix @ vector runs at a fraction of
/// FMA throughput. This runs several rows against `v` at once, each row with its own 4-lane
/// accumulator in exactly `dot_product`'s grouping, so the chains overlap and every output keeps
/// its bits. Blocks of 8 rows, then 4, then 2, so the rows left after the 8-row blocks (6 of a
/// 30-row `W`) overlap too. The matrix @ vector case goes through it, and `matmul_nt`'s tiles
/// keep its grouping, so a batched forward's rows stay bit-identical to the single-example forward.
fn dot_products_into(rows: &[f64], k: usize, v: &[f64], out: &mut [f64]) {
    let mut row = 0;
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            row = unsafe { dot_product_blocks_avx2_fma::<8>(rows, k, v, out, row) };
            row = unsafe { dot_product_blocks_avx2_fma::<4>(rows, k, v, out, row) };
            row = unsafe { dot_product_blocks_avx2_fma::<2>(rows, k, v, out, row) };
        }
    }
    // the rows left over after the last full block, or every row without AVX2
    for r in row..out.len() {
        out[r] = dot_product(&rows[r * k..(r + 1) * k], v);
    }
}

/// Fills `out[row..]` in blocks of `R` rows while a whole block fits; returns the first row left.
/// Safety: as `dot_product_rows_avx2_fma`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_blocks_avx2_fma<const R: usize>(
    rows: &[f64],
    k: usize,
    v: &[f64],
    out: &mut [f64],
    mut row: usize,
) -> usize {
    while row + R <= out.len() {
        let block = dot_product_rows_avx2_fma::<R>(&rows[row * k..(row + R) * k], k, v);
        out[row..row + R].copy_from_slice(&block);
        row += R;
    }
    row
}

/// `R` consecutive `k`-wide rows of `rows` against `v`: accumulator `r` is exactly
/// `dot_product_avx2_fma(rows[r], v)`'s `acc_vec` (the same loads, the same FMA operand order,
/// the same `i`), and the lanes combine and the tail finishes as there. Safety: only called after
/// `dot_products_into`'s runtime feature check; `rows` holds `R * k` values and `v` `k`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_rows_avx2_fma<const R: usize>(rows: &[f64], k: usize, v: &[f64]) -> [f64; R] {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    let rows_ptr = rows.as_ptr();
    let mut acc = [_mm256_setzero_pd(); R];
    let mut i = 0;
    while i + 4 <= k {
        let v_vec = _mm256_loadu_pd(v.as_ptr().add(i));
        for r in 0..R {
            acc[r] = _mm256_fmadd_pd(_mm256_loadu_pd(rows_ptr.add(r * k + i)), v_vec, acc[r]);
        }
        i += 4;
    }
    let mut out = [0.0; R];
    for r in 0..R {
        let mut lanes = [0.0f64; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), acc[r]);
        out[r] = dot_product_tail(&rows[r * k..(r + 1) * k], v, i, combine_lanes(lanes));
    }
    out
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
/// `tiled_row_range_avx2_fma` below.
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

/// `tiled_row_range` below in blocks of rows sized so one block of `a` is about 16 KB (1-row
/// blocks were 2-3x slower where `K` is in the hundreds, e.g. `(32, 512) @ (512, 5408)`), with
/// threaded row-splitting on top. Splits the output's row range across
/// `std::thread::scope` workers - safe without `'static` data (each worker borrows `a_data`/
/// `b_data` read-only and writes into its own disjoint slice of `out`, via `split_at_mut`) - only
/// once there's enough total work to pay for it (`matmul_thread_count`). Splitting by
/// row (not by `k` or `col`) needs no cross-thread reduction: each worker owns complete output
/// rows end to end, so results are bit-identical to the single-threaded path regardless of thread
/// count or scheduling - summation order per output row is unaffected by which thread computes it.
///
/// The flops check runs *before* anything else, including reading `available_parallelism_cached()`:
/// measured directly, `std::thread::available_parallelism()` itself costs ~50us per call (not
/// cached by the standard library, presumably a cgroup/proc filesystem read), which would have
/// silently dominated every one of this codebase's actual small per-call matmuls (already
/// measured in the tens of microseconds) if queried unconditionally on every dispatch. Cached
/// once behind a `OnceLock` and read only when there's already enough work to justify the
/// question.
///
/// `MODE` as in `tiled_row_range`: `OVERWRITE`, or `ADD`, where `out` holds `c` on entry and `c + a
/// @ b` on return.
fn matmul_2d<const MODE: u8>(a_data: &[f64], b_data: &[f64], out: &mut [f64], r1: usize, c1: usize, c2: usize) {
    const A_BLOCK_BYTES: usize = 16 * 1024;
    let rows_per_block = (A_BLOCK_BYTES / (c1 * std::mem::size_of::<f64>()).max(1)).max(1);
    for_each_row_range(out, r1, c2, r1 * c1 * c2, |chunk, row_start, row_end| {
        tiled_row_range::<MODE>(
            a_data,
            b_data,
            chunk,
            Panel::whole(row_start, row_end, c1, c2, rows_per_block),
        );
    });
}

/// `c + a @ b` for matrices, bit-identical to `matmul(a, b)` then an elementwise add: each
/// output's chain still starts from 0.0 and runs all of `k`, and only the finished chain is added
/// to `c`, one rounding as the separate add. Saves that add's full pass over the output, and one
/// output-sized array (`layer_accumulate_gradient_batch` at 32 x 5408 spent 140-160 us on the pass).
pub(crate) fn matmul_add(c: &RustArray, a: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    let (Shape::Matrix(r1, c1), Shape::Matrix(r2, c2)) = (a.shape, b.shape) else {
        return Err(shape_error(a.shape, b.shape));
    };
    if c1 != r2 {
        return Err(shape_error(a.shape, b.shape));
    }
    if c.shape != Shape::Matrix(r1, c2) {
        return Err(PyValueError::new_err(format!(
            "cannot add a product of shape {:?} to an array of shape {:?}",
            Shape::Matrix(r1, c2),
            c.shape
        )));
    }
    let mut out = c.data.clone();
    matmul_2d::<ADD>(&a.data, &b.data, &mut out, r1, c1, c2);
    Ok(RustArray::from_matrix(out, r1, c2))
}

static MAX_THREADS_OVERRIDE: AtomicUsize = AtomicUsize::new(0);
static THRESHOLD_FLOPS_OVERRIDE: AtomicUsize = AtomicUsize::new(0);

/// Test/benchmark hook: max threads (0 = default) and threshold in flops (0 = default) for every
/// threaded matmul (`matmul_2d`, `matmul_nt`, `matmul_narrow`), process-wide until reset with
/// `(0, 0)`. An overridden thread count isn't capped at the machine's parallelism, so a test can
/// run 8 threads anywhere. Can't change any output's value; that is what the tests check. A
/// function, not an env var, so one run can sweep settings.
#[pyfunction]
pub(crate) fn set_matmul_threading(max_threads: usize, threshold_flops: usize) {
    MAX_THREADS_OVERRIDE.store(max_threads, Ordering::Relaxed);
    THRESHOLD_FLOPS_OVERRIDE.store(threshold_flops, Ordering::Relaxed);
}

/// How many threads `for_each_row_range` uses for a `rows`-row product of `total_flops`: 1 below
/// the threshold, otherwise `min(available_parallelism, 8, rows)`, all or nothing. Measured on
/// this laptop (4 cores / 8 threads), not portable:
///
/// - 8M is where 8 threads start to beat 1 for `matmul_2d` and `matmul_nt` in isolation. The
///   old 4M threaded the MNIST conv mini-batch 32's dense tail (`32 x 5408`, 5.5M flops), which
///   made that epoch 11% slower: in training the other cores idle between calls, so every
///   threaded call starts on cold, clocked-down cores.
/// - Every product at or above 8M in the demos gained end to end at batch 512 (conv's 24.9M
///   ops and the 88.6M dense tail), or came out even (dense MNIST's 12M).
/// - No 2 or 4 threads: 2 never beat 1 up to 64M, and 4 only marginally beat 1 where 8 beat both.
///
/// No rows-per-thread floor either: products with few output rows and a long `k` gain the most
/// (conv's accumulate, 8 rows, halves on 8 threads at N = 512). With a floor of 32 rows per
/// thread, the MNIST conv mini-batch 512 epoch took 1.03-1.12 s against 0.84-0.88.
fn matmul_thread_count(rows: usize, total_flops: usize) -> usize {
    const THREADING_THRESHOLD_FLOPS: usize = 8_000_000;
    const MAX_THREADS: usize = 8;

    let threshold = match THRESHOLD_FLOPS_OVERRIDE.load(Ordering::Relaxed) {
        0 => THREADING_THRESHOLD_FLOPS,
        overridden => overridden,
    };
    if total_flops < threshold {
        return 1;
    }
    let max_threads = match MAX_THREADS_OVERRIDE.load(Ordering::Relaxed) {
        0 => available_parallelism_cached().min(MAX_THREADS),
        overridden => overridden,
    };
    max_threads.min(rows).max(1)
}

/// Test hook: the thread count the threaded matmuls would use for an `m x k @ k x n` product,
/// under the current policy and override, so a policy change shows up as a test failure.
#[pyfunction]
pub(crate) fn matmul_threads_for(m: usize, k: usize, n: usize) -> usize {
    matmul_thread_count(m, m * k * n)
}

/// The threading decision `matmul_thread_count` makes, shared by `matmul_2d`, `matmul_nt` and
/// `matmul_narrow`: calls `compute(chunk, row_start, row_end)` once on the whole `rows x cols`
/// output, or once per thread on disjoint row ranges. Each call owns complete output rows, so
/// the split can't change any output's value.
fn for_each_row_range<F>(out: &mut [f64], rows: usize, cols: usize, total_flops: usize, compute: F)
where
    F: Fn(&mut [f64], usize, usize) + Sync,
{
    let thread_count = matmul_thread_count(rows, total_flops);
    if thread_count <= 1 {
        compute(out, 0, rows);
        return;
    }

    let rows_per_thread = rows.div_ceil(thread_count);
    let compute = &compute;
    std::thread::scope(|scope| {
        let mut remaining_out = out;
        let mut row_start = 0;
        while row_start < rows {
            let row_end = (row_start + rows_per_thread).min(rows);
            let (chunk, rest) = remaining_out.split_at_mut((row_end - row_start) * cols);
            remaining_out = rest;
            scope.spawn(move || compute(chunk, row_start, row_end));
            row_start = row_end;
        }
    });
}

/// `a @ b.T` for `a` (`M, K`) and `b` (`N, K`), without materializing `b.T`: `out[m, n] =
/// dot_product(a[m], b[n])`, both rows contiguous. That is exactly the matrix @ vector case's
/// `matmul(b, a[m])` for each row `m`, so row `m` of the result is bit-identical to it, and a
/// batched forward agrees exactly with the single-example one. Rows of `a` go in register tiles
/// against rows of `b` (`matmul_nt_blocks_avx2_fma`), and the rows left over one at a time.
/// Threaded over `M` rows like `matmul_2d`.
pub(crate) fn matmul_nt(a: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    let (Shape::Matrix(m, k), Shape::Matrix(n, b_k)) = (a.shape, b.shape) else {
        return Err(PyValueError::new_err(format!(
            "matmul_nt requires two 2D arrays, got shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    };
    if k != b_k {
        return Err(PyValueError::new_err(format!(
            "cannot compute a @ b.T for shapes {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    let mut out = vec![0.0; m * n];
    for_each_row_range(&mut out, m, n, m * k * n, |chunk, row_start, row_end| {
        let mut row = row_start;
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                row = unsafe { matmul_nt_blocks_avx2_fma(&a.data, &b.data, chunk, row_start, row_end, k, n) };
            }
        }
        // the rows left over after the last full block, or every row without AVX2
        for row in row..row_end {
            let a_row = &a.data[row * k..(row + 1) * k];
            let out_row = &mut chunk[(row - row_start) * n..(row - row_start + 1) * n];
            dot_products_into(&b.data, k, a_row, out_row);
        }
    });
    Ok(RustArray::from_matrix(out, m, n))
}

/// `matmul_nt`'s register tile, 4 rows of `a` by 2 of `b`: 8 accumulators and 6 loads, in
/// AVX2's 16 registers. Measured against 2 x 4, 3 x 3 and 2 x 2 (dense `forward_batch` at 32 x
/// 5408, batch 32-512, and 30 x 784, batch 32 and 512), it was the fastest or tied at every shape.
const NT_A_ROWS: usize = 4;
const NT_B_ROWS: usize = 2;

/// `matmul_nt`'s rows `row_start..` in blocks of `NT_A_ROWS` rows of `a`, while a whole block
/// fits before `row_end`; returns the first row left. Each block runs against `b` `NT_B_ROWS`
/// rows at a time, so every load of `b` serves `NT_A_ROWS` outputs and every load of `a` serves
/// `NT_B_ROWS`: row by row, all of `b` (1.4 MB at 32 x 5408, past L2) streams in again for every
/// row of `a`. The `n % NT_B_ROWS` rows of `b` left over go through `dot_products_into`. Every
/// output is still one `dot_product` in its grouping, so the result is bit-identical to the row
/// by row loop. Safety: only called after `matmul_nt`'s runtime feature check; `a_data` holds
/// `row_end * k` values, `b_data` `n * k`, and `chunk` `(row_end - row_start) * n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_nt_blocks_avx2_fma(
    a_data: &[f64],
    b_data: &[f64],
    chunk: &mut [f64],
    row_start: usize,
    row_end: usize,
    k: usize,
    n: usize,
) -> usize {
    let mut row = row_start;
    while row + NT_A_ROWS <= row_end {
        let a_block = &a_data[row * k..(row + NT_A_ROWS) * k];
        let out_block = &mut chunk[(row - row_start) * n..(row - row_start + NT_A_ROWS) * n];
        let mut col = 0;
        while col + NT_B_ROWS <= n {
            let tile =
                dot_product_tile_avx2_fma::<NT_A_ROWS, NT_B_ROWS>(a_block, &b_data[col * k..(col + NT_B_ROWS) * k], k);
            for r in 0..NT_A_ROWS {
                out_block[r * n + col..r * n + col + NT_B_ROWS].copy_from_slice(&tile[r]);
            }
            col += NT_B_ROWS;
        }
        if col < n {
            for r in 0..NT_A_ROWS {
                dot_products_into(
                    &b_data[col * k..n * k],
                    k,
                    &a_block[r * k..(r + 1) * k],
                    &mut out_block[r * n + col..(r + 1) * n],
                );
            }
        }
        row += NT_A_ROWS;
    }
    row
}

/// `tile[r][c] = dot_product(b_rows[c], a_rows[r])` for `RA` consecutive `k`-wide rows of `a_rows` and `RB` of
/// `b_rows`, as `dot_products_into(b_rows, k, a_rows[r])` computes it: accumulator `(r, c)` is exactly
/// `dot_product_avx2_fma`'s `acc_vec` for that pair (the same loads, the same FMA operand order, the same `i`), and the
/// lanes combine and the tail finishes as there. `RA * RB` accumulators plus `RA + RB` loads have to fit AVX2's 16
/// registers. Safety: as `dot_product_rows_avx2_fma`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_tile_avx2_fma<const RA: usize, const RB: usize>(
    a_rows: &[f64],
    b_rows: &[f64],
    k: usize,
) -> [[f64; RB]; RA] {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    let a_ptr = a_rows.as_ptr();
    let b_ptr = b_rows.as_ptr();
    let mut acc = [[_mm256_setzero_pd(); RB]; RA];
    let mut i = 0;
    while i + 4 <= k {
        let b_vecs: [_; RB] = std::array::from_fn(|c| _mm256_loadu_pd(b_ptr.add(c * k + i)));
        for r in 0..RA {
            let a_vec = _mm256_loadu_pd(a_ptr.add(r * k + i));
            for c in 0..RB {
                acc[r][c] = _mm256_fmadd_pd(b_vecs[c], a_vec, acc[r][c]);
            }
        }
        i += 4;
    }
    let mut out = [[0.0; RB]; RA];
    for r in 0..RA {
        for c in 0..RB {
            let mut lanes = [0.0f64; 4];
            _mm256_storeu_pd(lanes.as_mut_ptr(), acc[r][c]);
            out[r][c] = dot_product_tail(
                &b_rows[c * k..(c + 1) * k],
                &a_rows[r * k..(r + 1) * k],
                i,
                combine_lanes(lanes),
            );
        }
    }
    out
}

/// Rows per block in `matmul_narrow`. Against 1 row, on one thread at N = 32: the second-conv
/// accumulate (`(8, N*P) @ (N*P, 72)`) about 40% faster, 4 rows ahead of 2 and 8; the
/// downstream (`(N*P, 8) @ (8, 72)`) 10-15% faster single-example and at 26x26x8 stride 2.
const NARROW_ROWS_PER_BLOCK: usize = 4;

/// Rows of `b` per slab in `matmul_long_k`. In an MNIST conv-conv mini-batch 32 epoch the
/// second-conv accumulate (`(8, 18432) @ (18432, 72)`, `b` 10.6 MB, past the 4 MB L3) went 5.4-5.9
/// → 2.6 ms a call (64-1024 all within 5%, 4096 3.2 ms), conv-pool-conv's (`b` 2.2 MB) 750 → 580-630
/// µs, and the single-example first conv 29 → 23 µs; small single-example shapes pay 0.1-0.5 µs.
const LONG_K_BLOCK: usize = 64;

static NARROW_ROWS_PER_BLOCK_OVERRIDE: AtomicUsize = AtomicUsize::new(0);
static LONG_K_BLOCK_OVERRIDE: AtomicUsize = AtomicUsize::new(0);

/// Benchmark hook: `matmul_narrow`'s rows per block and `matmul_long_k`'s slab rows (0 = the
/// default, `NARROW_ROWS_PER_BLOCK` and `LONG_K_BLOCK`), process-wide until reset with `(0, 0)`. A
/// slab of `K` rows or more runs `matmul_long_k` unblocked. Kept as instrumentation, so a later
/// change can re-measure these choices in one build; like `set_matmul_threading`, it can't change
/// any output's value, which the tests check.
#[pyfunction]
pub(crate) fn set_kernel_overrides(narrow_rows_per_block: usize, long_k_block: usize) {
    NARROW_ROWS_PER_BLOCK_OVERRIDE.store(narrow_rows_per_block, Ordering::Relaxed);
    LONG_K_BLOCK_OVERRIDE.store(long_k_block, Ordering::Relaxed);
}

fn overridden_or(overridden: &AtomicUsize, default: usize) -> usize {
    match overridden.load(Ordering::Relaxed) {
        0 => default,
        value => value,
    }
}

/// `a @ b` for `a` (`M, K`) and a narrow `b` (`K, N`, `N` a few dozen at most), the conv forward's
/// and downstream's matmul: the same kernel and bits as `matmul`'s matrix @ matrix case, in blocks
/// of `NARROW_ROWS_PER_BLOCK` rows. Both pass a tall `a` (`N * P` rows, `K` the fan-in or the
/// channel count). Threaded over `M` rows like `matmul_2d`, which doesn't change any output's
/// value.
pub(crate) fn matmul_narrow(a: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    let (m, k, n) = product_shape(a, b)?;
    let rows_per_block = overridden_or(&NARROW_ROWS_PER_BLOCK_OVERRIDE, NARROW_ROWS_PER_BLOCK);
    let mut out = vec![0.0; m * n];
    for_each_row_range(&mut out, m, n, m * k * n, |chunk, row_start, row_end| {
        let panel = Panel::whole(row_start, row_end, k, n, rows_per_block);
        tiled_row_range::<OVERWRITE>(&a.data, &b.data, chunk, panel);
    });
    Ok(RustArray::from_matrix(out, m, n))
}

/// `a @ b` for `a` (`M, K`) with few rows and a long `K`, and a narrow `b` (`K, N`): the conv
/// accumulate, `(O, N*P) @ (N*P, C*k*k)`, the same bits as `matmul`. Each output chain runs over
/// `K` in slabs of `LONG_K_BLOCK` rows of `b`, resuming from its stored partial (a stored double
/// is exact, so the chain's value is unchanged), so a slab serves every row and column tile from
/// cache before the next; one pass over all of `K` per tile read `b` from memory 4 times over
/// where it passed the L3. Threaded over 16-wide column chunks of `b`, not rows: each thread
/// computes every row for its columns into its own buffer, so `b` is still read about once (row
/// threading made every thread read all of it, and gained nothing from the slabs).
pub(crate) fn matmul_long_k(a: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    let (m, k, n) = product_shape(a, b)?;
    let k_block = overridden_or(&LONG_K_BLOCK_OVERRIDE, LONG_K_BLOCK);
    let chunks = n.div_ceil(16).max(1);
    let chunks_per_thread = chunks.div_ceil(matmul_thread_count(chunks, m * k * n));
    let column_ranges: Vec<(usize, usize)> = (0..chunks)
        .step_by(chunks_per_thread)
        .map(|chunk| (chunk * 16, ((chunk + chunks_per_thread) * 16).min(n)))
        .collect();
    let columns = |(col_start, col_end): (usize, usize)| -> Vec<f64> {
        let width = col_end - col_start;
        let mut part = vec![0.0; m * width];
        let mut k_start = 0;
        while k_start < k {
            let k_end = k_start.saturating_add(k_block).min(k);
            let panel = Panel {
                row_start: 0,
                row_end: m,
                rows_per_block: m,
                col_start,
                col_end,
                k_start,
                k_end,
                k,
                n,
                out_stride: width,
            };
            tiled_row_range::<RESUME>(&a.data, &b.data, &mut part, panel);
            k_start = k_end;
        }
        part
    };
    let parts: Vec<Vec<f64>> = if column_ranges.len() == 1 {
        vec![columns(column_ranges[0])]
    } else {
        std::thread::scope(|scope| {
            let columns = &columns;
            let handles: Vec<_> = column_ranges
                .iter()
                .map(|&range| scope.spawn(move || columns(range)))
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("matmul_long_k worker panicked"))
                .collect()
        })
    };
    let mut out = vec![0.0; m * n];
    for (&(col_start, col_end), part) in column_ranges.iter().zip(&parts) {
        let width = col_end - col_start;
        for row in 0..m {
            out[row * n + col_start..row * n + col_end].copy_from_slice(&part[row * width..(row + 1) * width]);
        }
    }
    Ok(RustArray::from_matrix(out, m, n))
}

fn product_shape(a: &RustArray, b: &RustArray) -> PyResult<(usize, usize, usize)> {
    let (Shape::Matrix(m, k), Shape::Matrix(b_k, n)) = (a.shape, b.shape) else {
        return Err(shape_error(a.shape, b.shape));
    };
    if k != b_k {
        return Err(shape_error(a.shape, b.shape));
    }
    Ok((m, k, n))
}

/// How `tiled_row_range` starts and stores each output's chain: from 0.0, stored over `out`; from
/// 0.0, added to the value in `out` (`c + chain`, the separate elementwise add's single rounding);
/// or continued from the partial chain in `out` (a slab of `k` after an earlier one).
pub(crate) const OVERWRITE: u8 = 0;
pub(crate) const ADD: u8 = 1;
pub(crate) const RESUME: u8 = 2;

/// The part of `a @ b` (`a` `M x K`, `b` `K x N`, both row-major) one `tiled_row_range` call
/// computes: rows `row_start..row_end`, columns `col_start..col_end`, and `k` over
/// `k_start..k_end`. Output `(row, col)` lives at `(row - row_start) * out_stride + col -
/// col_start` of the call's `out`. Only `RESUME` calls take part of the columns or of `k`; the
/// others must pass `Panel::whole`.
#[derive(Clone, Copy)]
pub(crate) struct Panel {
    pub(crate) row_start: usize,
    pub(crate) row_end: usize,
    pub(crate) rows_per_block: usize,
    pub(crate) col_start: usize,
    pub(crate) col_end: usize,
    pub(crate) k_start: usize,
    pub(crate) k_end: usize,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) out_stride: usize,
}

impl Panel {
    /// Rows `row_start..row_end`, every column and all of `k`, into a chunk that starts at row
    /// `row_start`.
    pub(crate) fn whole(row_start: usize, row_end: usize, k: usize, n: usize, rows_per_block: usize) -> Self {
        Panel {
            row_start,
            row_end,
            rows_per_block,
            col_start: 0,
            col_end: n,
            k_start: 0,
            k_end: k,
            k,
            n,
            out_stride: n,
        }
    }
}

/// `panel`'s outputs of `a @ b` into `out`. Every output is one FMA chain, `k` increasing, with
/// `a[m, k]` as the multiplier, started and stored as `MODE` says; the vector @ matrix case is
/// the one-row product. The AVX2 path holds a tile of output columns in registers across the
/// panel's `k` and stores it once. Accumulating a whole output row per `k` instead, as the vector
/// @ matrix case did before it came here, loads and stores the row `K` times, which dominated both
/// a narrow row (conv's `N` = 8, 2-3x slower) and a wide one (a 5408-wide row is 43 KB, past L1).
///
/// Blocked over rows: for each block of `rows_per_block` rows, each column tile runs over every
/// row of the block, so the tile's `K x 16` panel of `b` is read from cache by all of them. The
/// order of rows and tiles doesn't change any output's value.
pub(crate) fn tiled_row_range<const MODE: u8>(a_data: &[f64], b_data: &[f64], out: &mut [f64], panel: Panel) {
    debug_assert!(
        MODE == RESUME
            || (
                panel.col_start,
                panel.col_end,
                panel.k_start,
                panel.k_end,
                panel.out_stride
            ) == (0, panel.n, 0, panel.k, panel.n),
        "only RESUME takes part of a product"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { tiled_row_range_avx2_fma::<MODE>(a_data, b_data, out, panel) };
            return;
        }
    }
    tiled_row_range_scalar::<MODE>(a_data, b_data, out, panel);
}

fn tiled_row_range_scalar<const MODE: u8>(a_data: &[f64], b_data: &[f64], out: &mut [f64], panel: Panel) {
    let Panel { k, n, .. } = panel;
    for row in panel.row_start..panel.row_end {
        let a_row = &a_data[row * k + panel.k_start..row * k + panel.k_end];
        for col in panel.col_start..panel.col_end {
            let out_value = &mut out[(row - panel.row_start) * panel.out_stride + col - panel.col_start];
            let mut sum = if MODE == RESUME { *out_value } else { 0.0f64 };
            for (i, &a_value) in a_row.iter().enumerate() {
                sum = a_value.mul_add(b_data[(panel.k_start + i) * n + col], sum);
            }
            *out_value = if MODE == ADD { *out_value + sum } else { sum };
        }
    }
}

/// Rows per 16-column register tile in `tiled_row_range_avx2_fma`: each `k` loads the tile's `b`
/// row once for all of them, and their `4 * TILE_ROWS` independent FMA chains hide more of the FMA
/// latency than one row's 4 chains. A probe measured 2 rows 10-18% faster than 1 at short `k`
/// (`(32, 32) @ (32, 5408)`) and 7-31% at `k` = 512; 3 rows won only at `k` = 128.
const TILE_ROWS: usize = 2;

/// AVX2+FMA path: output columns in tiles of 16 (four 4-lane accumulators), then 4, then a
/// scalar tail, each tile running the panel's `k` before it is stored. The 16-wide tiles cover
/// `TILE_ROWS` rows of a block at once, and a block's leftover rows run them one row at a time.
/// Lane `j`'s accumulator is exactly `tiled_row_range_scalar`'s `sum` for that column, whichever
/// tile computes it. Safety: only called after `tiled_row_range`'s runtime feature check; every
/// pointer offset stays inside `a`'s rows `row_start..row_end`, `b`'s `k x n` data, or `out`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tiled_row_range_avx2_fma<const MODE: u8>(a_data: &[f64], b_data: &[f64], out: &mut [f64], panel: Panel) {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd};

    let Panel {
        row_start,
        row_end,
        rows_per_block,
        col_start,
        col_end,
        k_start,
        k_end,
        k,
        n,
        out_stride,
    } = panel;
    // only RESUME computes part of a product; the others take the whole panel's columns, `k` and
    // stride as known structure, which kept the whole-product callers 5-7% faster than runtime
    // offsets (single-example conv forward and downstream)
    let (col_start, col_end, k_start, k_end, out_stride) = if MODE == RESUME {
        (col_start, col_end, k_start, k_end, out_stride)
    } else {
        (0, n, 0, k, n)
    };
    let a_ptr = a_data.as_ptr();
    let b_ptr = b_data.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let out_at = move |row: usize, col: usize| out_ptr.add((row - row_start) * out_stride + col - col_start);
    let mut block_start = row_start;
    while block_start < row_end {
        let block_end = (block_start + rows_per_block).min(row_end);
        let mut col = col_start;
        while col + 16 <= col_end {
            let mut row = block_start;
            while row + TILE_ROWS <= block_end {
                let mut acc = [[chain_start::<MODE>(out_at(row, col)); 4]; TILE_ROWS];
                for (r, acc_row) in acc.iter_mut().enumerate() {
                    for (j, acc_lanes) in acc_row.iter_mut().enumerate() {
                        *acc_lanes = chain_start::<MODE>(out_at(row + r, col + 4 * j));
                    }
                }
                for kk in k_start..k_end {
                    let b_row = b_ptr.add(kk * n + col);
                    let b_tile = [
                        _mm256_loadu_pd(b_row),
                        _mm256_loadu_pd(b_row.add(4)),
                        _mm256_loadu_pd(b_row.add(8)),
                        _mm256_loadu_pd(b_row.add(12)),
                    ];
                    for r in 0..TILE_ROWS {
                        let a_vec = _mm256_set1_pd(*a_ptr.add((row + r) * k + kk));
                        for j in 0..4 {
                            acc[r][j] = _mm256_fmadd_pd(a_vec, b_tile[j], acc[r][j]);
                        }
                    }
                }
                for (r, acc_row) in acc.iter().enumerate() {
                    for (j, &acc_lanes) in acc_row.iter().enumerate() {
                        store_lanes::<MODE>(out_at(row + r, col + 4 * j), acc_lanes);
                    }
                }
                row += TILE_ROWS;
            }
            for row in row..block_end {
                let mut acc = [0, 4, 8, 12].map(|offset| chain_start::<MODE>(out_at(row, col + offset)));
                for (i, &a_value) in a_data[row * k + k_start..row * k + k_end].iter().enumerate() {
                    let a_vec = _mm256_set1_pd(a_value);
                    let b_row = b_ptr.add((k_start + i) * n + col);
                    acc[0] = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(b_row), acc[0]);
                    acc[1] = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(b_row.add(4)), acc[1]);
                    acc[2] = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(b_row.add(8)), acc[2]);
                    acc[3] = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(b_row.add(12)), acc[3]);
                }
                for (j, &acc_lanes) in acc.iter().enumerate() {
                    store_lanes::<MODE>(out_at(row, col + 4 * j), acc_lanes);
                }
            }
            col += 16;
        }
        while col + 4 <= col_end {
            for row in block_start..block_end {
                let mut acc = chain_start::<MODE>(out_at(row, col));
                for (i, &a_value) in a_data[row * k + k_start..row * k + k_end].iter().enumerate() {
                    acc = _mm256_fmadd_pd(
                        _mm256_set1_pd(a_value),
                        _mm256_loadu_pd(b_ptr.add((k_start + i) * n + col)),
                        acc,
                    );
                }
                store_lanes::<MODE>(out_at(row, col), acc);
            }
            col += 4;
        }
        while col < col_end {
            for row in block_start..block_end {
                let out_value = out_at(row, col);
                let mut sum = if MODE == RESUME { *out_value } else { 0.0f64 };
                for (i, &a_value) in a_data[row * k + k_start..row * k + k_end].iter().enumerate() {
                    sum = a_value.mul_add(*b_ptr.add((k_start + i) * n + col), sum);
                }
                *out_value = if MODE == ADD { *out_value + sum } else { sum };
            }
            col += 1;
        }
        block_start = block_end;
    }
}

/// 4 chains' starting values: the partials at `out` under `RESUME`, else 0.0. Safety: as
/// `tiled_row_range_avx2_fma`; `out` points at 4 values of its `out`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn chain_start<const MODE: u8>(out: *const f64) -> std::arch::x86_64::__m256d {
    if MODE == RESUME {
        std::arch::x86_64::_mm256_loadu_pd(out)
    } else {
        std::arch::x86_64::_mm256_setzero_pd()
    }
}

/// `tiled_row_range_avx2_fma`'s store of 4 chains, added to the 4 values at `out` under `ADD`.
/// Safety: as `tiled_row_range_avx2_fma`; `out` points at 4 values of its `out`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_lanes<const MODE: u8>(out: *mut f64, acc: std::arch::x86_64::__m256d) {
    use std::arch::x86_64::{_mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};

    if MODE == ADD {
        _mm256_storeu_pd(out, _mm256_add_pd(_mm256_loadu_pd(out), acc));
    } else {
        _mm256_storeu_pd(out, acc);
    }
}

fn shape_error(a_shape: Shape, b_shape: Shape) -> PyErr {
    PyValueError::new_err(format!("cannot matmul arrays of shape {:?} and {:?}", a_shape, b_shape))
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
