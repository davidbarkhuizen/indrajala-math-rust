//! A window-recurrent network's parts (indrajala-ml's window-recurrent adder workplan, stage 5):
//! one Rust function per pass of `StateInputArrayLayer`, `AttentionPoolingArrayLayer`,
//! `GRUArrayLayer` and `CodesArrayLayer` in
//! `indrajala_ml/model/layers/numpy/recurrent_array_layer.py`, and `concatenate_columns`, which
//! joins the windows' outputs. Every function takes a batch, one example per row; a window's `T`
//! tokens of `d` features are flat and token-major, index `t * d + j`, as in `tokens.rs`.
//!
//! ```text
//! state input   out_t = x_t + P c;  dc = (sum_t delta_t) P;  dP += (sum_t delta_t)^T c
//! pooling       s_t = q . x_t;  p = softmax(s);  u = sum_t p_t x_t
//!               dp_t = x_t . du;  ds = p * (dp - sum_t(dp_t p_t));  dq += sum_(n,t) ds_t x_t
//!               dx_t = p_t du + ds_t q
//! GRU           A = u W^T + b;  r, z = sigmoid(A_r,z + c U_r,z^T)
//!               h~ = tanh(A_h + (r * c) U_h^T);  h = z * c + (1 - z) * h~
//!               dA_h = dh * (1 - z) * (1 - h~ * h~);  d(rc) = dA_h U_h
//!               dA_r = d(rc) * c * r * (1 - r);  dA_z = dh * (c - h~) * z * (1 - z)
//!               dW += dA^T u;  db += sum_n dA;  dU_r,z += dA_r,z^T c;  dU_h += dA_h^T (r * c)
//!               du = dA W;  dc = (dh * z + d(rc) * r) + dA_r,z U_r,z
//! codes         i = argmin_k ||h - e_k||^2 (lowest k on a tie);  q = e_i
//!               dE_i += (-2 w) (h - q), row by row;  dh = dq + (2 w beta) (h - t)
//! ```
//!
//! where `t`, the commitment's target, is `q` unless a target is given (indrajala-ml's
//! code-consolidation workplan, D1: a duplicate code's state pulled to its representative's code).
//!
//! with `sigmoid(x) = 1 / (1 + exp(-x))` and `tanh(x) = 1 - 2 / (exp(2x) + 1)` (the workplan's W15),
//! `exp` Rust's `f64::exp`, as `exp` (`ufuncs.rs`) is. The products are the crate's (`matmul`,
//! `matmul_nt`, `matmul_add`, as the dense layer's ops call them: `affine_forward_batch`,
//! `linear_forward_batch`, `layer_downstream_batch`, `layer_accumulate_gradient_batch`,
//! `linear_accumulate_gradient_batch`), so their bits are those ops'; numpy's products are BLAS's,
//! not these. Every sum that isn't a product is a left fold from `0.0` in index order (numpy's
//! `np.cumsum`), the softmax is `array_softmax`, and every elementwise expression is numpy's,
//! grouped as written above, so the steps between products are numpy's by bits. The codes' search
//! and their backward pass have no product: they are numpy's by bits.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};
use crate::cr_exp::{exp_in_place, sigmoid_in_place};
use crate::fused::{affine_forward_batch, layer_accumulate_gradient_batch};
use crate::linalg::{matmul, matmul_add, matmul_nt};
use crate::ops::same_shape_elementwise;
use crate::ufuncs::array_softmax;

/// `(rows, cols)` of a matrix, refused for a vector.
pub(crate) fn matrix_shape(x: &RustArray, name: &str, context: &str) -> PyResult<(usize, usize)> {
    match x.shape {
        Shape::Matrix(rows, cols) => Ok((rows, cols)),
        shape => Err(PyValueError::new_err(format!(
            "{context} requires {name} as a matrix, got shape {shape:?}"
        ))),
    }
}

/// `x` refused unless of shape `expected`.
pub(crate) fn require_shape(x: &RustArray, expected: Shape, name: &str, context: &str) -> PyResult<()> {
    if x.shape != expected {
        return Err(PyValueError::new_err(format!(
            "{context} requires {name} of shape {expected:?}, got {:?}",
            x.shape
        )));
    }
    Ok(())
}

/// The features per token of `width` values in `tokens` tokens, refused unless it divides.
pub(crate) fn features_per_token(tokens: usize, width: usize, context: &str) -> PyResult<usize> {
    if tokens == 0 || !width.is_multiple_of(tokens) {
        return Err(PyValueError::new_err(format!(
            "{context} requires tokens x features values per example, {tokens} tokens, got {width}"
        )));
    }
    Ok(width / tokens)
}

/// `1 - 2 / (exp(2x) + 1)` for each of `values`, in place. The workplan's W15: through `exp`, as
/// `sigmoid`, so it agrees where `exp` does. A large `x` overflows `exp(2x)` to infinity and gives
/// exactly 1; a large negative `x` exactly -1. Doubling is exact, and each pass rounds value by
/// value, so the bits are the expression's.
fn tanh_in_place(values: &mut [f64]) {
    for v in values.iter_mut() {
        *v *= 2.0;
    }
    exp_in_place(values);
    for v in values.iter_mut() {
        *v = 1.0 - 2.0 / (*v + 1.0);
    }
}

/// Columns `start..end` of a `cols`-wide matrix, a copy.
pub(crate) fn columns(x: &RustArray, cols: usize, start: usize, end: usize) -> RustArray {
    let rows = x.data.len() / cols.max(1);
    let mut data = Vec::with_capacity(rows * (end - start));
    for row in x.data.chunks_exact(cols.max(1)) {
        data.extend_from_slice(&row[start..end]);
    }
    RustArray::from_matrix(data, rows, end - start)
}

/// Rows `start..end` of a `cols`-wide matrix, a copy.
fn row_block(x: &RustArray, cols: usize, start: usize, end: usize) -> RustArray {
    RustArray::from_matrix(x.data[start * cols..end * cols].to_vec(), end - start, cols)
}

/// Each example's `tokens` tokens of `features` summed, a left fold from `0.0` in token order:
/// `(rows, features)`.
fn token_sums(x: &RustArray, tokens: usize, features: usize) -> RustArray {
    let width = tokens * features;
    let rows = x.data.len() / width.max(1);
    let mut data = Vec::with_capacity(rows * features);
    for example in x.data.chunks_exact(width.max(1)) {
        let mut sums = vec![0.0; features];
        for token in example.chunks_exact(features.max(1)) {
            for (acc, &v) in sums.iter_mut().zip(token) {
                *acc += v;
            }
        }
        data.extend(sums);
    }
    RustArray::from_matrix(data, rows, features)
}

/// `StateInputArrayLayer.forward_batch`: `out_t = x_t + P c` for each of the `tokens` tokens of
/// each example, `x` `(N, T * d)`, `state` `(N, s)`, `p` `(d, s)`; `P c` is
/// `linear_forward_batch(p, state)`.
#[pyfunction]
pub fn state_input_forward(x: &RustArray, state: &RustArray, p: &RustArray, tokens: usize) -> PyResult<RustArray> {
    let context = "state_input_forward";
    let (rows, width) = matrix_shape(x, "x", context)?;
    let features = features_per_token(tokens, width, context)?;
    let (_, state_size) = matrix_shape(p, "p", context)?;
    require_shape(p, Shape::Matrix(features, state_size), "p", context)?;
    require_shape(state, Shape::Matrix(rows, state_size), "state", context)?;
    let projected = matmul_nt(state, p)?;
    let mut data = Vec::with_capacity(x.data.len());
    for (example, added) in x
        .data
        .chunks_exact(width.max(1))
        .zip(projected.data.chunks_exact(features.max(1)))
    {
        for token in example.chunks_exact(features.max(1)) {
            data.extend(token.iter().zip(added).map(|(&v, &a)| v + a));
        }
    }
    Ok(RustArray::from_matrix(data, rows, width))
}

/// `StateInputArrayLayer`'s backward pass: with `S` each example's delta summed over its tokens
/// (a left fold), the state's gradient `S @ p` and `p`'s gradient `grad_p + S^T @ state`. Returns
/// `(state_delta, grad_p)`.
#[pyfunction]
pub fn state_input_backward(
    delta: &RustArray,
    state: &RustArray,
    p: &RustArray,
    grad_p: &RustArray,
    tokens: usize,
) -> PyResult<(RustArray, RustArray)> {
    let context = "state_input_backward";
    let (rows, width) = matrix_shape(delta, "delta", context)?;
    let features = features_per_token(tokens, width, context)?;
    let (_, state_size) = matrix_shape(p, "p", context)?;
    require_shape(p, Shape::Matrix(features, state_size), "p", context)?;
    require_shape(grad_p, p.shape, "grad_p", context)?;
    require_shape(state, Shape::Matrix(rows, state_size), "state", context)?;
    let sums = token_sums(delta, tokens, features);
    Ok((matmul(&sums, p)?, matmul_add(grad_p, &sums.transpose(), state)?))
}

/// `(rows, features)` of a pooling's tokens `x`, `(N, T * d)`, and `q` checked to be `(d,)`.
fn pooling_shape(x: &RustArray, q: &RustArray, tokens: usize, context: &str) -> PyResult<(usize, usize)> {
    let (rows, width) = matrix_shape(x, "x", context)?;
    let features = features_per_token(tokens, width, context)?;
    require_shape(q, Shape::Vector(features), "q", context)?;
    Ok((rows, features))
}

/// `AttentionPoolingArrayLayer.forward_batch`: the scores `s_t = q . x_t` (`x`'s `(N * T, d)` rows
/// `@ q`), `p` their `array_softmax` per example, and `u = p @ x` per example, `(1, T) @ (T, d)`.
/// Returns `(u, p)`, `(N, d)` and `(N, T)`.
#[pyfunction]
pub fn attention_pooling_forward(x: &RustArray, q: &RustArray, tokens: usize) -> PyResult<(RustArray, RustArray)> {
    let (rows, features) = pooling_shape(x, q, tokens, "attention_pooling_forward")?;
    let token_rows = RustArray::from_matrix(x.data.clone(), rows * tokens, features);
    let scores = matmul(&token_rows, q)?;
    let p = array_softmax(&RustArray::from_matrix(scores.data, rows, tokens));
    let mut data = Vec::with_capacity(rows * features);
    for (weights, example) in p.data.chunks_exact(tokens).zip(x.data.chunks_exact(tokens * features)) {
        let weights = RustArray::from_matrix(weights.to_vec(), 1, tokens);
        let example = RustArray::from_matrix(example.to_vec(), tokens, features);
        data.extend(matmul(&weights, &example)?.data);
    }
    Ok((RustArray::from_matrix(data, rows, features), p))
}

/// `AttentionPoolingArrayLayer.backward_batch` after `attention_pooling_forward` gave `p`: `dp =
/// x @ du` per example, `(T, d) @ (d,)`; `ds = p * (dp - rowsum(dp * p))`; `q`'s gradient
/// `grad_q + colsum(ds_t x_t)` over the `N * T` rows; `dx_t = p_t du + ds_t q`. Every sum a left
/// fold. Returns `(dx, grad_q)`.
#[pyfunction]
pub fn attention_pooling_backward(
    x: &RustArray,
    p: &RustArray,
    q: &RustArray,
    du: &RustArray,
    grad_q: &RustArray,
    tokens: usize,
) -> PyResult<(RustArray, RustArray)> {
    let context = "attention_pooling_backward";
    let (rows, features) = pooling_shape(x, q, tokens, context)?;
    require_shape(p, Shape::Matrix(rows, tokens), "p", context)?;
    require_shape(du, Shape::Matrix(rows, features), "du", context)?;
    require_shape(grad_q, q.shape, "grad_q", context)?;
    let mut dx = Vec::with_capacity(x.data.len());
    let mut column_sums = vec![0.0; features];
    for ((example, weights), du_n) in x
        .data
        .chunks_exact(tokens * features)
        .zip(p.data.chunks_exact(tokens))
        .zip(du.data.chunks_exact(features))
    {
        let example_matrix = RustArray::from_matrix(example.to_vec(), tokens, features);
        let dp = matmul(&example_matrix, &RustArray::from_vector(du_n.to_vec()))?.data;
        let weighted = dp.iter().zip(weights).fold(0.0, |acc, (&d, &w)| acc + d * w);
        let ds: Vec<f64> = weights.iter().zip(&dp).map(|(&w, &d)| w * (d - weighted)).collect();
        for ((token, &w), &s) in example.chunks_exact(features).zip(weights).zip(&ds) {
            for (acc, &v) in column_sums.iter_mut().zip(token) {
                *acc += s * v;
            }
            dx.extend(du_n.iter().zip(&q.data).map(|(&g, &qj)| w * g + s * qj));
        }
    }
    let grad_q = same_shape_elementwise(&grad_q.data, &column_sums, |g, c| g + c);
    Ok((
        RustArray::from_matrix(dx, rows, tokens * features),
        RustArray::from_vector(grad_q),
    ))
}

/// The GRU's state size `s` from `u_weights`, `(3s, s)`, with `w` `(3s, d)` and `b` `(3s,)`
/// checked against it; returns `(s, d)`.
fn gru_sizes(w: &RustArray, u_weights: &RustArray, b: &RustArray, context: &str) -> PyResult<(usize, usize)> {
    let (rows, state_size) = matrix_shape(u_weights, "u_weights", context)?;
    if rows != 3 * state_size || state_size == 0 {
        return Err(PyValueError::new_err(format!(
            "{context} requires u_weights of shape (3s, s), s >= 1, got {:?}",
            u_weights.shape
        )));
    }
    let (w_rows, input_size) = matrix_shape(w, "w", context)?;
    if w_rows != rows {
        return Err(PyValueError::new_err(format!(
            "{context} requires w of shape ({rows}, d), got {:?}",
            w.shape
        )));
    }
    require_shape(b, Shape::Vector(rows), "b", context)?;
    Ok((state_size, input_size))
}

/// `GRUArrayLayer.forward_batch`: `u` `(N, d)` and `c` `(N, s)` to `h` `(N, s)`, the gates in the
/// order r, z, h, each a block of `s` rows of `w`, `u_weights` and `b`. `A` is
/// `affine_forward_batch(w, u, b)`, the state's parts `linear_forward_batch` of `u_weights`' row
/// blocks. Returns `(h, r, z, h_tilde, rc)`, what `gru_backward` reads.
#[pyfunction]
pub fn gru_forward(
    u: &RustArray,
    c: &RustArray,
    w: &RustArray,
    u_weights: &RustArray,
    b: &RustArray,
) -> PyResult<(RustArray, RustArray, RustArray, RustArray, RustArray)> {
    let context = "gru_forward";
    let (s, input_size) = gru_sizes(w, u_weights, b, context)?;
    let (rows, _) = matrix_shape(u, "u", context)?;
    require_shape(u, Shape::Matrix(rows, input_size), "u", context)?;
    require_shape(c, Shape::Matrix(rows, s), "c", context)?;
    let a = affine_forward_batch(w, u, b)?;
    let rz_state = matmul_nt(c, &row_block(u_weights, s, 0, 2 * s))?;
    let rz = same_shape_elementwise(&columns(&a, 3 * s, 0, 2 * s).data, &rz_state.data, |x, y| x + y);
    let mut r = Vec::with_capacity(rows * s);
    let mut z = Vec::with_capacity(rows * s);
    for gates in rz.chunks_exact(2 * s) {
        r.extend_from_slice(&gates[..s]);
        z.extend_from_slice(&gates[s..]);
    }
    sigmoid_in_place(&mut r);
    sigmoid_in_place(&mut z);
    let rc = RustArray::from_matrix(same_shape_elementwise(&r, &c.data, |x, y| x * y), rows, s);
    let h_state = matmul_nt(&rc, &row_block(u_weights, s, 2 * s, 3 * s))?;
    let mut h_tilde = same_shape_elementwise(&columns(&a, 3 * s, 2 * s, 3 * s).data, &h_state.data, |x, y| x + y);
    tanh_in_place(&mut h_tilde);
    let h = z
        .iter()
        .zip(&c.data)
        .zip(&h_tilde)
        .map(|((&zv, &cv), &hv)| zv * cv + (1.0 - zv) * hv)
        .collect();
    Ok((
        RustArray::from_matrix(h, rows, s),
        RustArray::from_matrix(r, rows, s),
        RustArray::from_matrix(z, rows, s),
        RustArray::from_matrix(h_tilde, rows, s),
        rc,
    ))
}

/// `GRUArrayLayer.backward_batch` after `gru_forward` gave `r`, `z`, `h_tilde` and `rc`: `h`'s
/// gradient `dh` `(N, s)` to `u`'s and `c`'s, and the parameters' gradients accumulated: `w`'s
/// and `b`'s as `layer_accumulate_gradient_batch(dA, u, ...)`, `u_weights`' row blocks as
/// `linear_accumulate_gradient_batch`. Returns `(du, dc, grad_w, grad_u, grad_b)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn gru_backward(
    dh: &RustArray,
    u: &RustArray,
    c: &RustArray,
    r: &RustArray,
    z: &RustArray,
    h_tilde: &RustArray,
    rc: &RustArray,
    w: &RustArray,
    u_weights: &RustArray,
    grad_w: &RustArray,
    grad_u: &RustArray,
    grad_b: &RustArray,
) -> PyResult<(RustArray, RustArray, RustArray, RustArray, RustArray)> {
    let context = "gru_backward";
    let (s, input_size) = gru_sizes(w, u_weights, grad_b, context)?;
    let (rows, _) = matrix_shape(dh, "dh", context)?;
    let state = Shape::Matrix(rows, s);
    for (x, name) in [
        (dh, "dh"),
        (c, "c"),
        (r, "r"),
        (z, "z"),
        (h_tilde, "h_tilde"),
        (rc, "rc"),
    ] {
        require_shape(x, state, name, context)?;
    }
    require_shape(u, Shape::Matrix(rows, input_size), "u", context)?;
    require_shape(grad_w, w.shape, "grad_w", context)?;
    require_shape(grad_u, u_weights.shape, "grad_u", context)?;

    let u_rz = row_block(u_weights, s, 0, 2 * s);
    let u_h = row_block(u_weights, s, 2 * s, 3 * s);
    let da_h: Vec<f64> = dh
        .data
        .iter()
        .zip(&z.data)
        .zip(&h_tilde.data)
        .map(|((&d, &zv), &hv)| d * (1.0 - zv) * (1.0 - hv * hv))
        .collect();
    let da_h = RustArray::from_matrix(da_h, rows, s);
    let d_rc = matmul(&da_h, &u_h)?;
    let mut da = Vec::with_capacity(rows * 3 * s);
    let mut da_rz = Vec::with_capacity(rows * 2 * s);
    for n in 0..rows {
        let row = n * s..(n + 1) * s;
        let gate_r = row
            .clone()
            .map(|i| d_rc.data[i] * c.data[i] * r.data[i] * (1.0 - r.data[i]));
        let gate_z = row
            .clone()
            .map(|i| dh.data[i] * (c.data[i] - h_tilde.data[i]) * z.data[i] * (1.0 - z.data[i]));
        let start = da_rz.len();
        da_rz.extend(gate_r);
        da_rz.extend(gate_z);
        da.extend_from_slice(&da_rz[start..]);
        da.extend_from_slice(&da_h.data[row]);
    }
    let da = RustArray::from_matrix(da, rows, 3 * s);
    let da_rz = RustArray::from_matrix(da_rz, rows, 2 * s);

    let (new_grad_w, new_grad_b) = layer_accumulate_gradient_batch(&da, u, grad_w, grad_b)?;
    let grad_u_rz = matmul_add(&row_block(grad_u, s, 0, 2 * s), &da_rz.transpose(), c)?;
    let grad_u_h = matmul_add(&row_block(grad_u, s, 2 * s, 3 * s), &da_h.transpose(), rc)?;
    let mut new_grad_u = grad_u_rz.data;
    new_grad_u.extend(grad_u_h.data);

    let du = matmul(&da, w)?;
    let dc_state = matmul(&da_rz, &u_rz)?;
    let dc = dh
        .data
        .iter()
        .zip(&z.data)
        .zip(d_rc.data.iter().zip(&r.data))
        .zip(&dc_state.data)
        .map(|(((&d, &zv), (&drc, &rv)), &cs)| (d * zv + drc * rv) + cs)
        .collect();
    Ok((
        du,
        RustArray::from_matrix(dc, rows, s),
        new_grad_w,
        RustArray::from_matrix(new_grad_u, 3 * s, s),
        new_grad_b,
    ))
}

/// `(count, size)` of the codes `e`, both 1 or more, and `h` checked to be `(N, size)`; returns
/// `(rows, count, size)`.
fn codes_shape(h: &RustArray, e: &RustArray, context: &str) -> PyResult<(usize, usize, usize)> {
    let (count, size) = matrix_shape(e, "e", context)?;
    if count == 0 || size == 0 {
        return Err(PyValueError::new_err(format!(
            "{context} requires e of shape (count, size), both 1 or more, got {:?}",
            e.shape
        )));
    }
    let (rows, _) = matrix_shape(h, "h", context)?;
    require_shape(h, Shape::Matrix(rows, size), "h", context)?;
    Ok((rows, count, size))
}

/// `CodesArrayLayer.forward_batch`: each row of `h` snapped to its nearest row of `e` by squared
/// distance, a left fold over the features of `(h - e)^2`; the lowest index on a tie, and the
/// first NaN distance if any, as `np.argmin`. Returns `(q, indices)`, the codes, a copy, and
/// their indices.
#[pyfunction]
pub fn codes_forward(h: &RustArray, e: &RustArray) -> PyResult<(RustArray, Vec<usize>)> {
    let (rows, _, size) = codes_shape(h, e, "codes_forward")?;
    let mut indices = Vec::with_capacity(rows);
    let mut q = Vec::with_capacity(h.data.len());
    for row in h.data.chunks_exact(size) {
        let mut best = 0;
        let mut best_distance = f64::INFINITY;
        for (k, code) in e.data.chunks_exact(size).enumerate() {
            let distance = row.iter().zip(code).fold(0.0, |acc, (&hv, &ev)| {
                let difference = hv - ev;
                acc + difference * difference
            });
            if distance.is_nan() {
                best = k;
                break;
            }
            if k == 0 || distance < best_distance {
                best = k;
                best_distance = distance;
            }
        }
        indices.push(best);
        q.extend_from_slice(&e.data[best * size..(best + 1) * size]);
    }
    Ok((RustArray::from_matrix(q, rows, size), indices))
}

/// `CodesArrayLayer.backward_batch` after `codes_forward` gave `q` and `indices`: with `D = h -
/// q`, `e`'s gradient `grad_e` with `(-2 weight) D` added row by row to the rows `indices` name,
/// in row order (`np.add.at`), and `h`'s gradient `dq + (2 weight commitment) (h - t)`, the
/// commitment's target `t` the rows of `target` `(N, size)`, or `q` without one (then `h - t` is
/// `D`, the same bits). Returns `(dh, grad_e)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (dq, h, q, indices, grad_e, weight, commitment, target = None))]
pub fn codes_backward(
    dq: &RustArray,
    h: &RustArray,
    q: &RustArray,
    indices: Vec<usize>,
    grad_e: &RustArray,
    weight: f64,
    commitment: f64,
    target: Option<&RustArray>,
) -> PyResult<(RustArray, RustArray)> {
    let context = "codes_backward";
    let (rows, count, size) = codes_shape(h, grad_e, context)?;
    require_shape(q, h.shape, "q", context)?;
    require_shape(dq, h.shape, "dq", context)?;
    if let Some(t) = target {
        require_shape(t, h.shape, "target", context)?;
    }
    if indices.len() != rows || indices.iter().any(|&i| i >= count) {
        return Err(PyValueError::new_err(format!(
            "{context} requires an index in [0, {count}) for each of the {rows} rows, got {indices:?}"
        )));
    }
    let differences = same_shape_elementwise(&h.data, &q.data, |hv, qv| hv - qv);
    let code_scale = -2.0 * weight;
    let state_scale = 2.0 * weight * commitment;
    let mut new_grad_e = grad_e.data.clone();
    for (&index, row) in indices.iter().zip(differences.chunks_exact(size)) {
        for (acc, &d) in new_grad_e[index * size..(index + 1) * size].iter_mut().zip(row) {
            *acc += code_scale * d;
        }
    }
    let dh = match target {
        None => same_shape_elementwise(&dq.data, &differences, |g, d| g + state_scale * d),
        Some(t) => {
            let pulled = same_shape_elementwise(&h.data, &t.data, |hv, tv| hv - tv);
            same_shape_elementwise(&dq.data, &pulled, |g, d| g + state_scale * d)
        }
    };
    Ok((
        RustArray::from_matrix(dh, rows, size),
        RustArray {
            data: new_grad_e,
            shape: grad_e.shape,
        },
    ))
}

/// A soft factor's temperature checked: finite and above 0.
pub(crate) fn require_temperature(temperature: f64, context: &str) -> PyResult<()> {
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(PyValueError::new_err(format!(
            "{context} requires a temperature above 0 and finite, got {temperature}"
        )));
    }
    Ok(())
}

/// The soft-to-hard quantiser's forward pass (indrajala-ml's commitment-control workplan, stage 3,
/// D6): with `d_k` each row's squared distance to code `k` (`codes_forward`'s left fold), the
/// weights `w = softmax(-d / temperature)`, `w_k = a_k / z`, `a_k = exp(-(d_k - m) / temperature)`,
/// `m` the row's smallest distance and `z` a left fold of the `a_k` in code order; and `q = sum_k
/// w_k e_k`, a left fold in code order. Returns `(q, weights, indices)`, `weights` `(N, count)` and
/// `indices` the nearest codes, as `codes_forward` gives them.
#[pyfunction]
pub fn soft_codes_forward(
    h: &RustArray,
    e: &RustArray,
    temperature: f64,
) -> PyResult<(RustArray, RustArray, Vec<usize>)> {
    let context = "soft_codes_forward";
    let (rows, count, size) = codes_shape(h, e, context)?;
    require_temperature(temperature, context)?;
    let (_, indices) = codes_forward(h, e)?;
    let mut weights = Vec::with_capacity(rows * count);
    let mut q = Vec::with_capacity(h.data.len());
    let mut distances = vec![0.0; count];
    for row in h.data.chunks_exact(size) {
        for (distance, code) in distances.iter_mut().zip(e.data.chunks_exact(size)) {
            *distance = row.iter().zip(code).fold(0.0, |acc, (&hv, &ev)| {
                let difference = hv - ev;
                acc + difference * difference
            });
        }
        let least = distances[1..]
            .iter()
            .fold(distances[0], |m, &d| if d < m { d } else { m });
        let start = weights.len();
        weights.extend(distances.iter().map(|&d| -(d - least) / temperature));
        let row_weights = &mut weights[start..];
        exp_in_place(row_weights);
        let total = row_weights[1..].iter().fold(row_weights[0], |acc, &a| acc + a);
        for w in row_weights.iter_mut() {
            *w /= total;
        }
        let row_weights = &weights[start..];
        for j in 0..size {
            let mut acc = row_weights[0] * e.data[j];
            for (k, &w) in row_weights.iter().enumerate().skip(1) {
                acc += w * e.data[k * size + j];
            }
            q.push(acc);
        }
    }
    Ok((
        RustArray::from_matrix(q, rows, size),
        RustArray::from_matrix(weights, rows, count),
        indices,
    ))
}

/// `soft_codes_forward`'s backward pass, from `g = dq`: per row, `c_k = w_k (g.e_k - g.q)` (the
/// dot products left folds over the features), `h`'s gradient `(2 / temperature) sum_k c_k (e_k -
/// q) + (2 weight commitment) (h - q)` (the sum a left fold in code order), the commitment held to
/// the soft `q`; and `e`'s gradient `grad_e` with, row by row in row order, code by code, `w_k g`,
/// then `(2 / temperature) c_k (h - e_k)`, then the soft codebook loss's `(-2 weight) w_k (h -
/// e_k)` added. With `straight`, `h`'s gradient is straight through, `g + (2 weight commitment) (h -
/// q)`, the softmax's term dropped (indrajala-ml's state formation workplan, D51: as the
/// temperature falls the soft path's gradient becomes the hard snap's); `e`'s as without it.
/// Returns `(dh, grad_e)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (dq, h, q, weights, e, grad_e, weight, commitment, temperature, straight = false))]
pub fn soft_codes_backward(
    dq: &RustArray,
    h: &RustArray,
    q: &RustArray,
    weights: &RustArray,
    e: &RustArray,
    grad_e: &RustArray,
    weight: f64,
    commitment: f64,
    temperature: f64,
    straight: bool,
) -> PyResult<(RustArray, RustArray)> {
    let context = "soft_codes_backward";
    let (rows, count, size) = codes_shape(h, e, context)?;
    require_temperature(temperature, context)?;
    require_shape(q, h.shape, "q", context)?;
    require_shape(dq, h.shape, "dq", context)?;
    require_shape(weights, Shape::Matrix(rows, count), "weights", context)?;
    require_shape(grad_e, e.shape, "grad_e", context)?;
    let scale = 2.0 / temperature;
    let state_scale = 2.0 * weight * commitment;
    let code_scale = -2.0 * weight;
    let dot = |a: &[f64], b: &[f64]| {
        a[1..]
            .iter()
            .zip(&b[1..])
            .fold(a[0] * b[0], |acc, (&x, &y)| acc + x * y)
    };
    let mut dh = Vec::with_capacity(h.data.len());
    let mut new_grad_e = grad_e.data.clone();
    let mut c = vec![0.0; count];
    for (((g, h_row), q_row), w_row) in dq
        .data
        .chunks_exact(size)
        .zip(h.data.chunks_exact(size))
        .zip(q.data.chunks_exact(size))
        .zip(weights.data.chunks_exact(count))
    {
        let gq = dot(g, q_row);
        for ((c_k, &w), code) in c.iter_mut().zip(w_row).zip(e.data.chunks_exact(size)) {
            *c_k = w * (dot(g, code) - gq);
        }
        for j in 0..size {
            let through = if straight {
                g[j]
            } else {
                let mut s = c[0] * (e.data[j] - q_row[j]);
                for (k, &c_k) in c.iter().enumerate().skip(1) {
                    s += c_k * (e.data[k * size + j] - q_row[j]);
                }
                scale * s
            };
            dh.push(through + state_scale * (h_row[j] - q_row[j]));
        }
        for (k, (&w, &c_k)) in w_row.iter().zip(&c).enumerate() {
            let (direct, soft, codebook) = (w, scale * c_k, code_scale * w);
            for j in 0..size {
                let at = k * size + j;
                let pulled = h_row[j] - e.data[at];
                new_grad_e[at] += direct * g[j];
                new_grad_e[at] += soft * pulled;
                new_grad_e[at] += codebook * pulled;
            }
        }
    }
    Ok((
        RustArray::from_matrix(dh, rows, size),
        RustArray {
            data: new_grad_e,
            shape: grad_e.shape,
        },
    ))
}

/// `np.concatenate(arrays, axis=1)`: matrices of one row count side by side, a copy.
#[pyfunction]
pub fn concatenate_columns(arrays: Vec<PyRef<'_, RustArray>>) -> PyResult<RustArray> {
    join_columns(
        &arrays.iter().map(|a| &**a).collect::<Vec<&RustArray>>(),
        "concatenate_columns",
    )
}

/// `concatenate_columns` over plain references, under `context`'s name.
pub(crate) fn join_columns(arrays: &[&RustArray], context: &str) -> PyResult<RustArray> {
    let Some(first) = arrays.first() else {
        return Err(PyValueError::new_err(format!(
            "{context} requires 1 or more arrays, got none"
        )));
    };
    let (rows, _) = matrix_shape(first, "every array", context)?;
    let mut widths = Vec::with_capacity(arrays.len());
    for array in arrays {
        let (array_rows, width) = matrix_shape(array, "every array", context)?;
        if array_rows != rows {
            return Err(PyValueError::new_err(format!(
                "{context} requires arrays of one row count, got {rows} and {array_rows}"
            )));
        }
        widths.push(width);
    }
    let total = widths.iter().sum();
    let mut data = Vec::with_capacity(rows * total);
    for row in 0..rows {
        for (array, &width) in arrays.iter().zip(&widths) {
            data.extend_from_slice(&array.data[row * width..(row + 1) * width]);
        }
    }
    Ok(RustArray::from_matrix(data, rows, total))
}
