//! Optimization 3 stage C measurement prototypes. Local branch only, never merged.
use pyo3::prelude::*;

use crate::array::RustArray;
use crate::conv::{channel_count, deltas_by_position, require_matrix, ConvGeometry};
use crate::linalg::{for_each_row_range, matmul};

fn im2col(x: &RustArray, g: &ConvGeometry, n: usize) -> Vec<f64> {
    let (p, k, fan_in) = (g.positions, g.kernel_size, g.fan_in);
    let mut cols = vec![0.0; n * p * fan_in];
    for example in 0..n {
        let input = &x.data[example * g.input_size..(example + 1) * g.input_size];
        for out_row in 0..g.out_height {
            for out_col in 0..g.out_width {
                let row = &mut cols[(example * p + out_row * g.out_width + out_col) * fan_in..][..fan_in];
                fill_row(g, input, out_row, out_col, k, row);
            }
        }
    }
    cols
}

#[inline]
fn fill_row(g: &ConvGeometry, input: &[f64], out_row: usize, out_col: usize, k: usize, row: &mut [f64]) {
    let mut column = 0;
    for c in 0..g.input_channels {
        for kr in 0..k {
            let start = g.input_index(c, out_row, out_col, kr, 0);
            row[column..column + k].copy_from_slice(&input[start..start + k]);
            column += k;
        }
    }
}

fn scatter(by_position: &[f64], b: &RustArray, n: usize, o: usize, p: usize) -> RustArray {
    let mut a = vec![0.0; n * o * p];
    for example in 0..n {
        for position in 0..p {
            let src = &by_position[(example * p + position) * o..][..o];
            for channel in 0..o {
                a[(example * o + channel) * p + position] = (src[channel] + b.data[channel]).max(0.0);
            }
        }
    }
    RustArray::from_matrix(a, n, o * p)
}

/// out[r, :] = sum_k rows[r, k] * w_t[k, :], k increasing from 0.0 with FMA - axpy_row's order,
/// with the O running sums held in registers.
fn rows_times_wt(rows: &[f64], row_start: usize, row_end: usize, fan_in: usize, w_t: &[f64], o: usize, out: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { rows_times_wt_avx2(rows, row_start, row_end, fan_in, w_t, o, out) };
            return;
        }
    }
    for r in row_start..row_end {
        let row = &rows[r * fan_in..][..fan_in];
        let out_row = &mut out[(r - row_start) * o..][..o];
        for oo in 0..o {
            let mut acc = 0.0f64;
            for k in 0..fan_in {
                acc = row[k].mul_add(w_t[k * o + oo], acc);
            }
            out_row[oo] = acc;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_times_wt_avx2(rows: &[f64], row_start: usize, row_end: usize, fan_in: usize, w_t: &[f64], o: usize, out: &mut [f64]) {
    use std::arch::x86_64::*;
    let wp = w_t.as_ptr();
    for r in row_start..row_end {
        let row = &rows[r * fan_in..][..fan_in];
        let out_ptr = out.as_mut_ptr().add((r - row_start) * o);
        let mut base = 0;
        while base + 16 <= o {
            let (mut a0, mut a1, mut a2, mut a3) = (_mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd());
            for k in 0..fan_in {
                let xv = _mm256_set1_pd(*row.get_unchecked(k));
                let w = wp.add(k * o + base);
                a0 = _mm256_fmadd_pd(xv, _mm256_loadu_pd(w), a0);
                a1 = _mm256_fmadd_pd(xv, _mm256_loadu_pd(w.add(4)), a1);
                a2 = _mm256_fmadd_pd(xv, _mm256_loadu_pd(w.add(8)), a2);
                a3 = _mm256_fmadd_pd(xv, _mm256_loadu_pd(w.add(12)), a3);
            }
            _mm256_storeu_pd(out_ptr.add(base), a0);
            _mm256_storeu_pd(out_ptr.add(base + 4), a1);
            _mm256_storeu_pd(out_ptr.add(base + 8), a2);
            _mm256_storeu_pd(out_ptr.add(base + 12), a3);
            base += 16;
        }
        while base + 4 <= o {
            let mut a0 = _mm256_setzero_pd();
            for k in 0..fan_in {
                a0 = _mm256_fmadd_pd(_mm256_set1_pd(*row.get_unchecked(k)), _mm256_loadu_pd(wp.add(k * o + base)), a0);
            }
            _mm256_storeu_pd(out_ptr.add(base), a0);
            base += 4;
        }
        for oo in base..o {
            let mut acc = 0.0f64;
            for k in 0..fan_in {
                acc = row[k].mul_add(*wp.add(k * o + oo), acc);
            }
            *out_ptr.add(oo) = acc;
        }
    }
}

/// C1: colsT (C*k*k, N*P), W @ colsT with matmul -> (O, N*P), then the (O, N, P) -> (N, O, P)
/// permute adding b and the ReLU. Returns (A, colsT).
#[pyfunction]
pub fn proto_conv_forward_c1(w: &RustArray, x: &RustArray, b: &RustArray, g: &ConvGeometry) -> PyResult<(RustArray, RustArray)> {
    let o = channel_count(w, g, "c1")?;
    let n = require_matrix(x, None, g.input_size, "c1")?;
    let (p, k, fan_in) = (g.positions, g.kernel_size, g.fan_in);
    let np = n * p;
    let mut cols_t = vec![0.0; fan_in * np];
    for example in 0..n {
        let input = &x.data[example * g.input_size..(example + 1) * g.input_size];
        for c in 0..g.input_channels {
            for kr in 0..k {
                for kc in 0..k {
                    let dst = &mut cols_t[((c * k + kr) * k + kc) * np + example * p..][..p];
                    for out_row in 0..g.out_height {
                        for out_col in 0..g.out_width {
                            dst[out_row * g.out_width + out_col] = input[g.input_index(c, out_row, out_col, kr, kc)];
                        }
                    }
                }
            }
        }
    }
    let cols_t = RustArray::from_matrix(cols_t, fan_in, np);
    let by_channel = matmul(w, &cols_t)?; // (O, N*P)
    let mut a = vec![0.0; n * o * p];
    for channel in 0..o {
        for example in 0..n {
            let src = &by_channel.data[channel * np + example * p..][..p];
            let dst = &mut a[(example * o + channel) * p..][..p];
            let bias = b.data[channel];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = (s + bias).max(0.0);
            }
        }
    }
    Ok((RustArray::from_matrix(a, n, o * p), cols_t))
}

/// C2: im2col as now (cols kept for the backward pass), then the register-blocked small-O kernel
/// instead of matmul, threaded like matmul. Returns (A, cols).
#[pyfunction]
pub fn proto_conv_forward_c2(w: &RustArray, x: &RustArray, b: &RustArray, g: &ConvGeometry) -> PyResult<(RustArray, RustArray)> {
    let o = channel_count(w, g, "c2")?;
    let n = require_matrix(x, None, g.input_size, "c2")?;
    let (p, fan_in) = (g.positions, g.fan_in);
    let cols = im2col(x, g, n);
    let w_t = w.transpose();
    let mut by_position = vec![0.0; n * p * o];
    for_each_row_range(&mut by_position, n * p, o, n * p * fan_in * o, |chunk, start, end| {
        rows_times_wt(&cols, start, end, fan_in, &w_t.data, o, chunk);
    });
    Ok((scatter(&by_position, b, n, o, p), RustArray::from_matrix(cols, n * p, fan_in)))
}

/// C2-direct: no cols; one scratch im2col row per position, then the same kernel. Returns A.
#[pyfunction]
pub fn proto_conv_forward_c2_direct(w: &RustArray, x: &RustArray, b: &RustArray, g: &ConvGeometry) -> PyResult<RustArray> {
    let o = channel_count(w, g, "c2d")?;
    let n = require_matrix(x, None, g.input_size, "c2d")?;
    let (p, k, fan_in) = (g.positions, g.kernel_size, g.fan_in);
    let w_t = w.transpose();
    let mut row = vec![0.0; fan_in];
    let mut by_position = vec![0.0; n * p * o];
    for example in 0..n {
        let input = &x.data[example * g.input_size..(example + 1) * g.input_size];
        for out_row in 0..g.out_height {
            for out_col in 0..g.out_width {
                fill_row(g, input, out_row, out_col, k, &mut row);
                let r = example * p + out_row * g.out_width + out_col;
                rows_times_wt(&row, 0, 1, fan_in, &w_t.data, o, &mut by_position[r * o..(r + 1) * o]);
            }
        }
    }
    Ok(scatter(&by_position, b, n, o, p))
}

fn grad_b_update(delta: &RustArray, grad_b: &RustArray, n: usize, o: usize, p: usize) -> RustArray {
    // the same per-channel sum as conv_accumulate_gradient_batch: sequential over (example, position)
    let mut by_channel = vec![0.0; o * n * p];
    for example in 0..n {
        for channel in 0..o {
            let src = &delta.data[(example * o + channel) * p..][..p];
            by_channel[channel * n * p + example * p..][..p].copy_from_slice(src);
        }
    }
    RustArray::from_vector(
        grad_b.data.iter().enumerate()
            .map(|(c, &gb)| gb + by_channel[c * n * p..(c + 1) * n * p].iter().sum::<f64>())
            .collect(),
    )
}

/// C1 backward, option 1: transpose colsT back to cols, then the existing D @ cols.
#[pyfunction]
pub fn proto_conv_accumulate_c1_copy(delta: &RustArray, cols_t: &RustArray, grad_w: &RustArray, grad_b: &RustArray, g: &ConvGeometry) -> PyResult<(RustArray, RustArray)> {
    let cols = cols_t.transpose();
    crate::conv::conv_accumulate_gradient_batch(delta, &cols, grad_w, grad_b, g)
}

/// C1 backward, option 2: update.T = colsT @ D_by_position (C*k*k, O), each element the same
/// sequential FMA chain over (example, position) as D @ cols, then a small transpose.
#[pyfunction]
pub fn proto_conv_accumulate_c1_tn(delta: &RustArray, cols_t: &RustArray, grad_w: &RustArray, grad_b: &RustArray, g: &ConvGeometry) -> PyResult<(RustArray, RustArray)> {
    let o = channel_count(grad_w, g, "c1tn")?;
    let p = g.positions;
    let n = require_matrix(delta, None, o * p, "c1tn")?;
    let update_t = matmul(cols_t, &deltas_by_position(delta, n, o, p))?; // (C*k*k, O)
    let new_grad_w = grad_w.combine_with_array(&update_t.transpose(), |a, u| a + u, "add")?;
    Ok((new_grad_w, grad_b_update(delta, grad_b, n, o, p)))
}
