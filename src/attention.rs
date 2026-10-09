//! Multi-head self-attention (indrajala-ml's multi-head attention workplan, D4): one Rust function
//! per pass of `AttentionArrayLayer` in `indrajala_ml/model/layers/numpy/attention_array_layer.py`,
//! each one call, so a pass crosses the Python/Rust boundary once. Each pass is a thin wrapper
//! that validates the shapes, builds an `AttentionOptions` and composes the crate-internal blocks,
//! project, attend and combine, in indrajala-ml's README's expressions (Layer norm and attention)
//! and their grouping, with `h` heads of `d_k` features and `[i]` head `i`'s block:
//!
//! ```text
//! project   Q = X Wq^T + bq;  K = X Wk^T + bk;  V = X Wv^T + bv           (T, h·d_k) each
//! attend    S[i] = (Q[i] K[i]^T) / s;  P[i] = softmax_rows(S[i]);  H[i] = P[i] V[i]
//! combine   H = [H[0] ... H[h-1]];  out = H Wo^T + bo
//! backward  dH = delta Wo
//!           dP[i] = dH[i] V[i]^T;  dV[i] = P[i]^T dH[i];  dS[i] = P[i] * (dP[i] - rowsum(dP[i] * P[i]))
//!           dQ[i] = (dS[i] K[i]) / s;  dK[i] = (dS[i]^T Q[i]) / s
//!           dX = (dQ Wq + dK Wk) + dV Wv
//! ```
//!
//! with `s = sqrt(d_k)`. Each block is composed of the crate's existing ops, so it has their bits:
//! project, combine and the parameter gradients are a dense layer's on the `(N * T, d)` and
//! `(N * T, h·d_k)` rows (`affine_forward_batch`, `layer_downstream_batch`,
//! `layer_accumulate_gradient_batch`), one product over all heads; attend's products, per example
//! and head, are `@` (`matmul`) on `(T, d_k)` and `(T, T)` matrices, a transpose copied first as
//! Python's `a @ b.T` copies it; the row softmax is `array_softmax`; `rowsum` is a left fold from
//! `0.0` over each row. At one head with `d_k = d` every expression is the single head's.
//!
//! The caches are packed: `Q`, `K`, `V`, `H` and their gradients `(N * T, h·d_k)`, head `i` in
//! columns `i·d_k..`, and `P` `(N * T, h·T)`, head `i` in columns `i·T..`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};
use crate::fused::{affine_forward_batch, layer_accumulate_gradient_batch, layer_downstream_batch};
use crate::linalg::matmul;
use crate::ufuncs::array_softmax;

/// What attend reads besides `Q`, `K` and `V`: the head count and each head's width. Masks and
/// dropout on `P` are fields to come (the workplan's Extension points).
#[derive(Clone, Copy)]
struct AttentionOptions {
    heads: usize,
    key_size: usize,
}

impl AttentionOptions {
    /// `heads` heads across the projections' `width = h·d_k` columns.
    fn new(heads: usize, width: usize, context: &str) -> PyResult<Self> {
        if heads == 0 || width == 0 || !width.is_multiple_of(heads) {
            return Err(PyValueError::new_err(format!(
                "{context} requires heads >= 1 dividing the projections' width {width}, got heads={heads}"
            )));
        }
        Ok(AttentionOptions {
            heads,
            key_size: width / heads,
        })
    }

    /// `h·d_k`, the projections' width.
    fn width(&self) -> usize {
        self.heads * self.key_size
    }

    /// `sqrt(d_k)`, computed once and divided by, never multiplied by its reciprocal.
    fn scale(&self) -> f64 {
        (self.key_size as f64).sqrt()
    }
}

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

fn require_shape(array: &RustArray, shape: Shape, name: &str, context: &str) -> PyResult<()> {
    if array.shape != shape {
        return Err(PyValueError::new_err(format!(
            "{context} requires {name} of shape {shape:?}, got {:?}",
            array.shape
        )));
    }
    Ok(())
}

/// `wq`, `wk`, `wv` `(h·d_k, d)` and `wo` `(d, h·d_k)`, `d` and `width` given.
fn require_weights(weights: [&RustArray; 4], d: usize, width: usize, context: &str) -> PyResult<()> {
    let [wq, wk, wv, wo] = weights;
    for (w, name) in [(wq, "wq"), (wk, "wk"), (wv, "wv")] {
        require_shape(w, Shape::Matrix(width, d), name, context)?;
    }
    require_shape(wo, Shape::Matrix(d, width), "wo", context)
}

/// `(d, options)`, `d` the token width (`bo`'s length) and the projections' width `bq`'s: every
/// projection's weight and bias shaped to them.
fn shapes(projections: &Projections, heads: usize, context: &str) -> PyResult<(usize, AttentionOptions)> {
    let (d, width) = (projections.bo.data.len(), projections.bq.data.len());
    if d == 0 {
        return Err(PyValueError::new_err(format!("{context} requires a non-empty bo")));
    }
    let options = AttentionOptions::new(heads, width, context)?;
    let weights = [projections.wq, projections.wk, projections.wv, projections.wo];
    require_weights(weights, d, width, context)?;
    for (b, name) in [(projections.bq, "bq"), (projections.bk, "bk"), (projections.bv, "bv")] {
        require_shape(b, Shape::Vector(width), name, context)?;
    }
    require_shape(projections.bo, Shape::Vector(d), "bo", context)?;
    Ok((d, options))
}

/// `(examples, tokens)` of `x`, 1D (one example) or 2D, `T * d` values per example.
fn examples_and_tokens(x: &RustArray, d: usize, name: &str, context: &str) -> PyResult<(usize, usize)> {
    let (rows, width) = match x.shape {
        Shape::Vector(n) => (1, n),
        Shape::Matrix(rows, n) => (rows, n),
    };
    if d == 0 || width == 0 || !width.is_multiple_of(d) {
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

fn columns(m: &RustArray) -> usize {
    match m.shape {
        Shape::Matrix(_, width) => width,
        Shape::Vector(width) => width,
    }
}

/// Example `n`'s head `i`: the `(T, width)` block of an `(N * T, h * width)` matrix at rows
/// `n·T..` and columns `i·width..`, copied.
fn head(m: &RustArray, n: usize, i: usize, tokens: usize, width: usize) -> RustArray {
    let stride = columns(m);
    let mut block = Vec::with_capacity(tokens * width);
    for row in m.data[n * tokens * stride..(n + 1) * tokens * stride].chunks_exact(stride) {
        block.extend_from_slice(&row[i * width..(i + 1) * width]);
    }
    RustArray::from_matrix(block, tokens, width)
}

/// `head`'s inverse: `block` written into example `n`'s head `i` of `m`'s values, `stride` wide.
fn set_head(m: &mut [f64], stride: usize, n: usize, i: usize, block: &RustArray) {
    let width = columns(block);
    let tokens = block.data.len() / width;
    let rows = m[n * tokens * stride..(n + 1) * tokens * stride].chunks_exact_mut(stride);
    for (row, values) in rows.zip(block.data.chunks_exact(width)) {
        row[i * width..(i + 1) * width].copy_from_slice(values);
    }
}

/// Each value divided by `s`.
fn divided(m: RustArray, s: f64) -> RustArray {
    RustArray {
        data: m.data.iter().map(|&v| v / s).collect(),
        shape: m.shape,
    }
}

// project, attend, combine: the forward pass's blocks

/// `Q`, `K`, `V` from the `(N * T, d)` input rows: each the product, then the bias, over all heads.
fn project(rows: &RustArray, projections: &Projections) -> PyResult<(RustArray, RustArray, RustArray)> {
    let q = affine_forward_batch(projections.wq, rows, projections.bq)?;
    let k = affine_forward_batch(projections.wk, rows, projections.bk)?;
    let v = affine_forward_batch(projections.wv, rows, projections.bv)?;
    Ok((q, k, v))
}

/// Per example and head, the weights and the weighted sum: `(p, h)`, `p` `(N * T, h·T)` and `h`
/// `(N * T, h·d_k)`, from the packed `(N * T, h·d_k)` `q`, `k`, `v`.
fn attend_forward(
    q: &RustArray,
    k: &RustArray,
    v: &RustArray,
    tokens: usize,
    options: &AttentionOptions,
) -> PyResult<(RustArray, RustArray)> {
    let (heads, d_k, width, s) = (options.heads, options.key_size, options.width(), options.scale());
    let rows = q.data.len() / width;
    let mut p = vec![0.0; rows * heads * tokens];
    let mut h = vec![0.0; rows * width];
    for n in 0..rows / tokens {
        for i in 0..heads {
            let (q_ni, k_ni, v_ni) = (
                head(q, n, i, tokens, d_k),
                head(k, n, i, tokens, d_k),
                head(v, n, i, tokens, d_k),
            );
            let scores = divided(matmul(&q_ni, &k_ni.transpose())?, s);
            let p_ni = array_softmax(&scores);
            set_head(&mut h, width, n, i, &matmul(&p_ni, &v_ni)?);
            set_head(&mut p, heads * tokens, n, i, &p_ni);
        }
    }
    Ok((
        RustArray::from_matrix(p, rows, heads * tokens),
        RustArray::from_matrix(h, rows, width),
    ))
}

/// `H`, the heads side by side, through the output projection: one product over all heads.
fn combine(h: &RustArray, projections: &Projections) -> PyResult<RustArray> {
    affine_forward_batch(projections.wo, h, projections.bo)
}

// the backward pass's blocks, last first

/// `dH = delta Wo`, `(N * T, h·d_k)`, from the `(N * T, d)` delta rows.
fn combine_backward(delta: &RustArray, wo: &RustArray) -> PyResult<RustArray> {
    layer_downstream_batch(wo, delta)
}

/// Per example and head, the softmax's and the products' gradients: `(dq, dk, dv)`, each
/// `(N * T, h·d_k)`, from the forward pass's packed caches and `dh`.
fn attend_backward(
    caches: [&RustArray; 4],
    dh: &RustArray,
    tokens: usize,
    options: &AttentionOptions,
) -> PyResult<(RustArray, RustArray, RustArray)> {
    let [q, k, v, p] = caches;
    let (heads, d_k, width, s) = (options.heads, options.key_size, options.width(), options.scale());
    let rows = dh.data.len() / width;
    let mut dq = vec![0.0; rows * width];
    let mut dk = vec![0.0; rows * width];
    let mut dv = vec![0.0; rows * width];
    for n in 0..rows / tokens {
        for i in 0..heads {
            let (q_ni, k_ni, v_ni) = (
                head(q, n, i, tokens, d_k),
                head(k, n, i, tokens, d_k),
                head(v, n, i, tokens, d_k),
            );
            let (p_ni, dh_ni) = (head(p, n, i, tokens, tokens), head(dh, n, i, tokens, d_k));
            let dp = matmul(&dh_ni, &v_ni.transpose())?;
            set_head(&mut dv, width, n, i, &matmul(&p_ni.transpose(), &dh_ni)?);
            let mut ds = Vec::with_capacity(tokens * tokens);
            for (dp_row, p_row) in dp.data.chunks_exact(tokens).zip(p_ni.data.chunks_exact(tokens)) {
                let r = dp_row.iter().zip(p_row).fold(0.0, |acc, (&dpv, &pv)| acc + dpv * pv);
                ds.extend(dp_row.iter().zip(p_row).map(|(&dpv, &pv)| pv * (dpv - r)));
            }
            let ds = RustArray::from_matrix(ds, tokens, tokens);
            set_head(&mut dq, width, n, i, &divided(matmul(&ds, &k_ni)?, s));
            set_head(&mut dk, width, n, i, &divided(matmul(&ds.transpose(), &q_ni)?, s));
        }
    }
    Ok((
        RustArray::from_matrix(dq, rows, width),
        RustArray::from_matrix(dk, rows, width),
        RustArray::from_matrix(dv, rows, width),
    ))
}

/// `dX = (dQ Wq + dK Wk) + dV Wv`, `(N * T, d)`: three products over all heads, summed in this order.
fn project_backward(gradients: [&RustArray; 3], weights: [&RustArray; 3]) -> PyResult<RustArray> {
    let [dq, dk, dv] = gradients;
    let [wq, wk, wv] = weights;
    let dx_q = layer_downstream_batch(wq, dq)?;
    let dx_k = layer_downstream_batch(wk, dk)?;
    let dx_v = layer_downstream_batch(wv, dv)?;
    let dx = dx_q
        .data
        .iter()
        .zip(&dx_k.data)
        .zip(&dx_v.data)
        .map(|((&a, &b), &c)| (a + b) + c)
        .collect();
    Ok(RustArray {
        data: dx,
        shape: dx_q.shape,
    })
}

/// The forward pass: `(a, q, k, v, p, h)`, `a` in `x`'s shape, the caches as `(N * T, h·d_k)`
/// rows and `p` as `(N * T, h·T)`.
type ForwardResult = (RustArray, RustArray, RustArray, RustArray, RustArray, RustArray);

fn forward(x: &RustArray, projections: &Projections, heads: usize, context: &str) -> PyResult<ForwardResult> {
    let (d, options) = shapes(projections, heads, context)?;
    let (_, tokens) = examples_and_tokens(x, d, "x", context)?;
    let (q, k, v) = project(&token_rows(x, d), projections)?;
    let (p, h) = attend_forward(&q, &k, &v, tokens, &options)?;
    let out = combine(&h, projections)?;
    let a = RustArray {
        data: out.data,
        shape: x.shape,
    };
    Ok((a, q, k, v, p, h))
}

/// `AttentionArrayLayer.forward`: one example, `x` 1D of `T * d` values, `d` `bo`'s length, in
/// `heads` heads across `bq`'s length. Returns `(a, q, k, v, p, h)`: `a` 1D, and what the backward
/// pass reads, `q`, `k`, `v` and `h` `(T, h·d_k)`, `p` `(T, h·T)`.
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
    heads: usize,
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
    forward(x, &projections, heads, context)
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
    heads: usize,
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
    forward(x, &projections, heads, context)
}

/// `AttentionArrayLayer._backward`: from `delta_batch` (`dl/dout`, 2D `batch, T * d`, or 1D for
/// one example, whose caches are `attention_forward`'s) and the forward pass's `q`, `k`, `v` and
/// `p`, in `heads` heads across `wq`'s rows, returns `(dx, dq, dk, dv)`: `dx`, the downstream, in
/// `delta_batch`'s shape, and `dq`, `dk`, `dv` as `(N * T, h·d_k)` rows, which the gradients read.
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
    heads: usize,
) -> PyResult<(RustArray, RustArray, RustArray, RustArray)> {
    let context = "attention_downstream_batch";
    let (width, d) = match wq.shape {
        Shape::Matrix(width, d) => (width, d),
        Shape::Vector(d) => (d, 1),
    };
    let options = AttentionOptions::new(heads, width, context)?;
    require_weights([wq, wk, wv, wo], d, width, context)?;
    let (examples, tokens) = examples_and_tokens(delta_batch, d, "delta_batch", context)?;
    let rows = examples * tokens;
    for (cache, name) in [(q, "q"), (k, "k"), (v, "v")] {
        require_shape(cache, Shape::Matrix(rows, width), name, context)?;
    }
    require_shape(p, Shape::Matrix(rows, heads * tokens), "p", context)?;

    let dh = combine_backward(&token_rows(delta_batch, d), wo)?;
    let (dq, dk, dv) = attend_backward([q, k, v, p], &dh, tokens, &options)?;
    let dx = project_backward([&dq, &dk, &dv], [wq, wk, wv])?;
    let dx = RustArray {
        data: dx.data,
        shape: delta_batch.shape,
    };
    Ok((dx, dq, dk, dv))
}

/// `AttentionArrayLayer.accumulate_gradient_batch`: each projection's `(grad_W, grad_b)` as a
/// dense layer's, `grad_W += delta^T X` and `grad_b += sum(delta)` over the `(N * T)` rows, with
/// `dq`, `dk`, `dv` (`attention_downstream_batch`'s) against `x`, and `delta_batch` against `h`
/// (the forward pass's), `delta_batch` and `x` 2D or, for one example, 1D. Head-free: each is one
/// product over all heads. Returns the eight updated gradients in the parameters' order.
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
    let (d, width) = (grad_bo.data.len(), grad_bq.data.len());
    let (examples, tokens) = examples_and_tokens(delta_batch, d, "delta_batch", context)?;
    let rows = examples * tokens;
    require_shape(x, delta_batch.shape, "x", context)?;
    for (cache, name) in [(h, "h"), (dq, "dq"), (dk, "dk"), (dv, "dv")] {
        require_shape(cache, Shape::Matrix(rows, width), name, context)?;
    }
    let x_rows = token_rows(x, d);
    let delta = token_rows(delta_batch, d);
    let (grad_wq, grad_bq) = layer_accumulate_gradient_batch(dq, &x_rows, grad_wq, grad_bq)?;
    let (grad_wk, grad_bk) = layer_accumulate_gradient_batch(dk, &x_rows, grad_wk, grad_bk)?;
    let (grad_wv, grad_bv) = layer_accumulate_gradient_batch(dv, &x_rows, grad_wv, grad_bv)?;
    let (grad_wo, grad_bo) = layer_accumulate_gradient_batch(&delta, h, grad_wo, grad_bo)?;
    Ok((grad_wq, grad_bq, grad_wk, grad_bk, grad_wv, grad_bv, grad_wo, grad_bo))
}
