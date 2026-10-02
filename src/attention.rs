//! Single-head self-attention (indrajala-ml's layer-norm and attention workplan, D6 and D9): one
//! Rust function per pass of `AttentionArrayLayer` in `indrajala_ml/model/attention_array_layer.py`,
//! each one call, so a pass crosses the Python/Rust boundary once. The expressions are
//! indrajala-ml's README's (Layer norm and attention), in their grouping:
//!
//! ```text
//! Q = X Wq^T + bq;  K = X Wk^T + bk;  V = X Wv^T + bv;  S = (Q K^T) / s;  P = softmax_rows(S)
//! H = P V;  out = H Wo^T + bo
//! dH = delta Wo;  dP = dH V^T;  dV = P^T dH;  dS = P * (dP - rowsum(dP * P))
//! dQ = (dS K) / s;  dK = (dS^T Q) / s;  dX = (dQ Wq + dK Wk) + dV Wv
//! ```
//!
//! with `s = sqrt(d)`. Each is composed of the crate's existing ops, so it has their bits: the
//! projections and the parameter gradients are a dense layer's on the `(N * T, d)` rows
//! (`affine_forward_batch`, `layer_downstream_batch`, `layer_accumulate_gradient_batch`); the
//! products between activations, per example, are `@` (`matmul`) on `(T, d)` and `(T, T)`
//! matrices, a transpose copied first as Python's `a @ b.T` copies it; the row softmax is
//! `array_softmax`; `rowsum` is a left fold from `0.0` over each row. A head count would be one
//! more argument; this is the single head.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};
use crate::fused::{affine_forward_batch, layer_accumulate_gradient_batch, layer_downstream_batch};
use crate::linalg::matmul;
use crate::ufuncs::array_softmax;

/// The weights and biases in their order, which is also their draw order.
struct Projections<'a> {
    wq: &'a RustArray,
    bq: &'a RustArray,
    wk: &'a RustArray,
    bk: &'a RustArray,
    wv: &'a RustArray,
    bv: &'a RustArray,
    wo: &'a RustArray,
    bo: &'a RustArray,
}

/// `d`, the token width: every weight `(d, d)` and every bias `(d,)`.
fn features(projections: &Projections, context: &str) -> PyResult<usize> {
    let d = projections.bq.data.len();
    let weights = [projections.wq, projections.wk, projections.wv, projections.wo];
    let biases = [projections.bq, projections.bk, projections.bv, projections.bo];
    let ok = d > 0
        && weights.iter().all(|w| w.shape == Shape::Matrix(d, d))
        && biases.iter().all(|b| b.shape == Shape::Vector(d));
    if !ok {
        return Err(PyValueError::new_err(format!(
            "{context} requires every weight of shape ({d}, {d}) and every bias of shape ({d},), bq's length"
        )));
    }
    Ok(d)
}

/// `(examples, tokens)` of `x`, 1D (one example) or 2D, `T * d` values per example.
fn examples_and_tokens(x: &RustArray, d: usize, name: &str, context: &str) -> PyResult<(usize, usize)> {
    let (rows, width) = match x.shape {
        Shape::Vector(n) => (1, n),
        Shape::Matrix(rows, n) => (rows, n),
    };
    if width == 0 || !width.is_multiple_of(d) {
        return Err(PyValueError::new_err(format!(
            "{context} requires {name} of whole tokens of {d} features per example, got shape {:?}",
            x.shape
        )));
    }
    Ok((rows, width / d))
}

/// `x`'s rows of `d`: `(N * T, d)`, the same values.
fn token_rows(x: &RustArray, d: usize) -> RustArray {
    RustArray::from_matrix(x.data.clone(), x.data.len() / d, d)
}

/// Example `n`'s `(T, width)` block of an `(N * T, width)` matrix, copied.
fn example(m: &RustArray, n: usize, tokens: usize) -> RustArray {
    let width = match m.shape {
        Shape::Matrix(_, width) => width,
        Shape::Vector(width) => width,
    };
    let block = tokens * width;
    RustArray::from_matrix(m.data[n * block..(n + 1) * block].to_vec(), tokens, width)
}

/// Each value divided by `s`.
fn divided(m: RustArray, s: f64) -> RustArray {
    RustArray {
        data: m.data.iter().map(|&v| v / s).collect(),
        shape: m.shape,
    }
}

fn require_shape(array: &RustArray, shape: Shape, name: &str, context: &str) -> PyResult<()> {
    if array.shape != shape {
        return Err(PyValueError::new_err(format!(
            "{context} requires {name} of shape {shape:?}, got {:?}",
            array.shape
        )));
    }
    Ok(())
}

/// The forward pass: `(a, q, k, v, p, h)`, `a` in `x`'s shape, the caches as `(N * T, d)` rows
/// and `p` as `(N * T, T)`.
type ForwardResult = (RustArray, RustArray, RustArray, RustArray, RustArray, RustArray);

fn forward(x: &RustArray, projections: &Projections, context: &str) -> PyResult<ForwardResult> {
    let d = features(projections, context)?;
    let (examples, tokens) = examples_and_tokens(x, d, "x", context)?;
    let rows = token_rows(x, d);
    // each the product, then the bias
    let q = affine_forward_batch(projections.wq, &rows, projections.bq)?;
    let k = affine_forward_batch(projections.wk, &rows, projections.bk)?;
    let v = affine_forward_batch(projections.wv, &rows, projections.bv)?;
    let s = (d as f64).sqrt();
    let mut p = Vec::with_capacity(examples * tokens * tokens);
    let mut h = Vec::with_capacity(examples * tokens * d);
    for n in 0..examples {
        let (q_n, k_n, v_n) = (example(&q, n, tokens), example(&k, n, tokens), example(&v, n, tokens));
        let scores = divided(matmul(&q_n, &k_n.transpose())?, s);
        let p_n = array_softmax(&scores);
        h.extend(matmul(&p_n, &v_n)?.data);
        p.extend(p_n.data);
    }
    let p = RustArray::from_matrix(p, examples * tokens, tokens);
    let h = RustArray::from_matrix(h, examples * tokens, d);
    let out = affine_forward_batch(projections.wo, &h, projections.bo)?;
    let a = RustArray {
        data: out.data,
        shape: x.shape,
    };
    Ok((a, q, k, v, p, h))
}

/// `AttentionArrayLayer.forward`: one example, `x` 1D of `T * d` values, `d` the projections'
/// width. Returns `(a, q, k, v, p, h)`: `a` 1D, and what the backward pass reads, `q`, `k`, `v`
/// and `h` `(T, d)`, `p` `(T, T)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn attention_forward(
    x: &RustArray,
    wq: &RustArray,
    bq: &RustArray,
    wk: &RustArray,
    bk: &RustArray,
    wv: &RustArray,
    bv: &RustArray,
    wo: &RustArray,
    bo: &RustArray,
) -> PyResult<ForwardResult> {
    let context = "attention_forward";
    if !matches!(x.shape, Shape::Vector(_)) {
        return Err(PyValueError::new_err(format!(
            "{context} requires a 1D x, got shape {:?}",
            x.shape
        )));
    }
    let projections = Projections {
        wq,
        bq,
        wk,
        bk,
        wv,
        bv,
        wo,
        bo,
    };
    forward(x, &projections, context)
}

/// `AttentionArrayLayer.forward_batch`: `x` 2D (`batch, T * d`), each example
/// `attention_forward`'s bits. Returns `(a, q, k, v, p, h)`, the caches over the batch's `(N * T)`
/// rows, examples then tokens.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn attention_forward_batch(
    x: &RustArray,
    wq: &RustArray,
    bq: &RustArray,
    wk: &RustArray,
    bk: &RustArray,
    wv: &RustArray,
    bv: &RustArray,
    wo: &RustArray,
    bo: &RustArray,
) -> PyResult<ForwardResult> {
    let context = "attention_forward_batch";
    if !matches!(x.shape, Shape::Matrix(_, _)) {
        return Err(PyValueError::new_err(format!(
            "{context} requires a 2D x, got shape {:?}",
            x.shape
        )));
    }
    let projections = Projections {
        wq,
        bq,
        wk,
        bk,
        wv,
        bv,
        wo,
        bo,
    };
    forward(x, &projections, context)
}

/// `AttentionArrayLayer._backward`: from `delta_batch` (`dl/dout`, 2D `batch, T * d`) and the
/// forward pass's `q`, `k`, `v` and `p`, returns `(dx, dq, dk, dv)`: `dx`, the downstream, in
/// `delta_batch`'s shape, and `dq`, `dk`, `dv` as `(N * T, d)` rows, which the gradients read.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn attention_downstream_batch(
    delta_batch: &RustArray,
    wq: &RustArray,
    wk: &RustArray,
    wv: &RustArray,
    wo: &RustArray,
    q: &RustArray,
    k: &RustArray,
    v: &RustArray,
    p: &RustArray,
) -> PyResult<(RustArray, RustArray, RustArray, RustArray)> {
    let context = "attention_downstream_batch";
    if !matches!(delta_batch.shape, Shape::Matrix(_, _)) {
        return Err(PyValueError::new_err(format!(
            "{context} requires a 2D delta_batch, got shape {:?}",
            delta_batch.shape
        )));
    }
    let d = match wq.shape {
        Shape::Matrix(d, _) => d,
        Shape::Vector(d) => d,
    };
    for (w, name) in [(wq, "wq"), (wk, "wk"), (wv, "wv"), (wo, "wo")] {
        require_shape(w, Shape::Matrix(d, d), name, context)?;
    }
    let (examples, tokens) = examples_and_tokens(delta_batch, d, "delta_batch", context)?;
    let rows = examples * tokens;
    for (cache, name) in [(q, "q"), (k, "k"), (v, "v")] {
        require_shape(cache, Shape::Matrix(rows, d), name, context)?;
    }
    require_shape(p, Shape::Matrix(rows, tokens), "p", context)?;

    let delta = token_rows(delta_batch, d);
    let dh = layer_downstream_batch(wo, &delta)?;
    let s = (d as f64).sqrt();
    let mut dq = Vec::with_capacity(rows * d);
    let mut dk = Vec::with_capacity(rows * d);
    let mut dv = Vec::with_capacity(rows * d);
    for n in 0..examples {
        let (q_n, k_n, v_n) = (example(q, n, tokens), example(k, n, tokens), example(v, n, tokens));
        let (p_n, dh_n) = (example(p, n, tokens), example(&dh, n, tokens));
        let dp = matmul(&dh_n, &v_n.transpose())?;
        dv.extend(matmul(&p_n.transpose(), &dh_n)?.data);
        let mut ds = Vec::with_capacity(tokens * tokens);
        for (dp_row, p_row) in dp.data.chunks_exact(tokens).zip(p_n.data.chunks_exact(tokens)) {
            let r = dp_row.iter().zip(p_row).fold(0.0, |acc, (&dpv, &pv)| acc + dpv * pv);
            ds.extend(dp_row.iter().zip(p_row).map(|(&dpv, &pv)| pv * (dpv - r)));
        }
        let ds = RustArray::from_matrix(ds, tokens, tokens);
        dq.extend(divided(matmul(&ds, &k_n)?, s).data);
        dk.extend(divided(matmul(&ds.transpose(), &q_n)?, s).data);
    }
    let dq = RustArray::from_matrix(dq, rows, d);
    let dk = RustArray::from_matrix(dk, rows, d);
    let dv = RustArray::from_matrix(dv, rows, d);
    // three products, summed in this order
    let dx_q = layer_downstream_batch(wq, &dq)?;
    let dx_k = layer_downstream_batch(wk, &dk)?;
    let dx_v = layer_downstream_batch(wv, &dv)?;
    let dx: Vec<f64> = dx_q
        .data
        .iter()
        .zip(&dx_k.data)
        .zip(&dx_v.data)
        .map(|((&a, &b), &c)| (a + b) + c)
        .collect();
    let dx = RustArray {
        data: dx,
        shape: delta_batch.shape,
    };
    Ok((dx, dq, dk, dv))
}

/// `AttentionArrayLayer.accumulate_gradient_batch`: each projection's `(grad_W, grad_b)` as a
/// dense layer's, `grad_W += delta^T X` and `grad_b += sum(delta)` over the `(N * T)` rows, with
/// `dq`, `dk`, `dv` (`attention_downstream_batch`'s) against `x`, and `delta_batch` against `h`
/// (the forward pass's). Returns the eight updated gradients in the parameters' order.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
pub fn attention_accumulate_gradient_batch(
    delta_batch: &RustArray,
    x: &RustArray,
    h: &RustArray,
    dq: &RustArray,
    dk: &RustArray,
    dv: &RustArray,
    grad_wq: &RustArray,
    grad_bq: &RustArray,
    grad_wk: &RustArray,
    grad_bk: &RustArray,
    grad_wv: &RustArray,
    grad_bv: &RustArray,
    grad_wo: &RustArray,
    grad_bo: &RustArray,
) -> PyResult<(
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
)> {
    let context = "attention_accumulate_gradient_batch";
    let d = grad_bq.data.len();
    let (examples, tokens) = examples_and_tokens(delta_batch, d.max(1), "delta_batch", context)?;
    let rows = examples * tokens;
    require_shape(x, delta_batch.shape, "x", context)?;
    for (cache, name) in [(h, "h"), (dq, "dq"), (dk, "dk"), (dv, "dv")] {
        require_shape(cache, Shape::Matrix(rows, d), name, context)?;
    }
    let x_rows = token_rows(x, d);
    let delta = token_rows(delta_batch, d);
    let (grad_wq, grad_bq) = layer_accumulate_gradient_batch(dq, &x_rows, grad_wq, grad_bq)?;
    let (grad_wk, grad_bk) = layer_accumulate_gradient_batch(dk, &x_rows, grad_wk, grad_bk)?;
    let (grad_wv, grad_bv) = layer_accumulate_gradient_batch(dv, &x_rows, grad_wv, grad_bv)?;
    let (grad_wo, grad_bo) = layer_accumulate_gradient_batch(&delta, h, grad_wo, grad_bo)?;
    Ok((grad_wq, grad_bq, grad_wk, grad_bk, grad_wv, grad_bv, grad_wo, grad_bo))
}
