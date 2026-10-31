//! Multi-head self-attention (indrajala-ml's multi-head attention workplan, D4): one Rust function
//! per pass of `AttentionArrayLayer` in `indrajala_ml/model/layers/numpy/attention_array_layer.py`,
//! each one call, so a pass crosses the Python/Rust boundary once. Each pass is a thin wrapper
//! that validates the shapes, builds an `AttentionOptions` and composes the crate-internal blocks,
//! project, attend and combine, in indrajala-ml's README's expressions (Layer norm and attention)
//! and their grouping, with `h` heads of `d_k` features and `[i]` head `i`'s block:
//!
//! ```text
//! project   Q = X Wq^T + bq;  K = X Wk^T + bk;  V = X Wv^T + bv           (T, h·d_k) each
//! attend    S[i] = (Q[i] K[i]^T) / s;  S[i]_tj = -inf for j > t if causal
//!           P[i] = softmax_rows(S[i]);  H[i] = P[i] V[i]
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
//! A causal mask (indrajala-ml's sequence task workplan, D7) sets each score `S_tj` with `j > t`
//! to `-inf` after the scale and before the row max, as numpy's `AttentionArrayLayer` does: the row
//! max comes from the unmasked scores (the diagonal never is), `exp(-inf)` is exactly 0, so a
//! masked `P_tj` is exactly 0 and so is its `dS_tj`. The backward pass reads `P` and needs no mask,
//! so only the forward ops take `causal`. An unmasked pass computes what it did before the mask.
//!
//! Dropout on `P` (indrajala-ml's attention-dropout workplan, D3-D5): a training pass with
//! `dropout > 0` draws a 0/1 mask `M[i]` per example and head from the network's `Generator`, `T²`
//! values each in (example, head, query, key) order, numpy's row-major `(N, h, T, T)` draw, every
//! entry drawn, the causally masked ones included; `M_tj = u_tj >= dropout`. Then
//!
//! ```text
//! attend    P~[i] = P[i] * M[i] / keep;  H[i] = P~[i] V[i]
//! backward  dP[i] = (dH[i] V[i]^T) * M[i] / keep;  dV[i] = P~[i]^T dH[i]
//! ```
//!
//! with `keep = 1 - dropout`, the softmax's backward reading the undropped `P`. The forward ops
//! return `M`, packed as `P`, which the backward op takes; `P~` is recomputed from `P` and `M`, the
//! same elementwise expression and so the same bits. Without `training`, and at `dropout = 0`,
//! nothing is drawn, the mask is `None` and every op computes what it did before dropout.
//!
//! Rotary positions (indrajala-ml's rotary-positions workplan, D3-D5): a pass given `cos` and
//! `sin`, `(T, d_k / 2)` each, row `t` token `t`'s angles, ends project by rotating each head's
//! `Q` and `K`, after the biases, feature `j < d_k / 2` paired with `j + d_k / 2`:
//!
//! ```text
//! project   q'_j = q_j c - q_{j+d_k/2} s;  q'_{j+d_k/2} = q_j s + q_{j+d_k/2} c    (k likewise)
//! backward  dq_j = dq'_j c + dq'_{j+d_k/2} s;  dq_{j+d_k/2} = dq'_{j+d_k/2} c - dq'_j s
//! ```
//!
//! each entry two products and a sum, numpy's elementwise expressions, so the same bits. The
//! caches `q` and `k` are the rotated ones, which attend reads; the downstream op rotates `dQ` and
//! `dK` back before project's products and returns them so, the gradients of the unrotated `Q` and
//! `K`, which the gradient op takes as before, so it needs no table. `V` isn't rotated. Without a
//! table every op computes what it did before rotation.
//!
//! The caches are packed: `Q`, `K`, `V`, `H` and their gradients `(N * T, h·d_k)`, head `i` in
//! columns `i·d_k..`, and `P` `(N * T, h·T)`, head `i` in columns `i·T..`.

use crate::array::{Array, Shape};
use crate::fused::{affine_forward_batch, layer_accumulate_gradient_batch, layer_downstream_batch};
use crate::generator::Generator;
use crate::linalg::matmul;
use crate::ufuncs::array_softmax;
use crate::Error;

/// What attend reads besides `Q`, `K` and `V`: the head count, each head's width, whether the
/// scores are masked causally and the drop probability of each weight in training.
#[derive(Clone, Copy)]
struct AttentionOptions {
    heads: usize,
    key_size: usize,
    causal: bool,
    dropout: f64,
}

impl AttentionOptions {
    /// `heads` heads across the projections' `width = h·d_k` columns, `dropout` in `[0, 1)`.
    fn new(heads: usize, width: usize, causal: bool, dropout: f64, context: &str) -> Result<Self, Error> {
        if heads == 0 || width == 0 || !width.is_multiple_of(heads) {
            return Err(Error::invalid(format!(
                "{context} requires heads >= 1 dividing the projections' width {width}, got heads={heads}"
            )));
        }
        if !(0.0..1.0).contains(&dropout) {
            return Err(Error::invalid(format!(
                "{context} requires a dropout in [0, 1), got {dropout}"
            )));
        }
        Ok(AttentionOptions {
            heads,
            key_size: width / heads,
            causal,
            dropout,
        })
    }

    /// `1 - dropout`, each kept weight's divisor.
    fn keep(&self) -> f64 {
        1.0 - self.dropout
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
    wq: &'a Array,
    bq: &'a Array,
    wk: &'a Array,
    bk: &'a Array,
    wv: &'a Array,
    bv: &'a Array,
    wo: &'a Array,
    bo: &'a Array,
}

/// A rotary pass's table (indrajala-ml's rotary-positions workplan, D4, D5): `cos` and `sin`,
/// `(T, d_k / 2)` each, row `t` token `t`'s angles, computed by the caller.
#[derive(Clone, Copy)]
struct Rotation<'a> {
    cos: &'a Array,
    sin: &'a Array,
}

impl<'a> Rotation<'a> {
    /// The table from the ops' `cos` and `sin` keywords: both or neither, each `(T, d_k / 2)` with
    /// `d_k` even.
    fn new(
        cos: Option<&'a Array>,
        sin: Option<&'a Array>,
        tokens: usize,
        options: &AttentionOptions,
        context: &str,
    ) -> Result<Option<Self>, Error> {
        let (cos, sin) = match (cos, sin) {
            (None, None) => return Ok(None),
            (Some(cos), Some(sin)) => (cos, sin),
            _ => {
                return Err(Error::invalid(format!(
                    "{context} requires both cos and sin to rotate, or neither"
                )));
            }
        };
        if !options.key_size.is_multiple_of(2) {
            return Err(Error::invalid(format!(
                "{context} requires an even head size to rotate, got d_k={}",
                options.key_size
            )));
        }
        let shape = Shape::Matrix(tokens, options.key_size / 2);
        require_shape(cos, shape, "cos", context)?;
        require_shape(sin, shape, "sin", context)?;
        Ok(Some(Rotation { cos, sin }))
    }

    /// The packed `(N * T, h·d_k)` rows `m`, each head's features `j` and `j + d_k / 2` rotated by
    /// the row's token's angles: `(a c - b s, a s + b c)`; or back, `inverse`, by the transpose:
    /// `(a c + b s, b c - a s)`. Each entry two products and a sum, as numpy's elementwise
    /// expressions, so the same bits.
    fn rotated(&self, m: &Array, tokens: usize, options: &AttentionOptions, inverse: bool) -> Array {
        let (d_k, width) = (options.key_size, options.width());
        let half = d_k / 2;
        let mut data = m.data.clone();
        for (r, (row, out)) in m.data.chunks_exact(width).zip(data.chunks_exact_mut(width)).enumerate() {
            let t = r % tokens;
            let (cos, sin) = (
                &self.cos.data[t * half..(t + 1) * half],
                &self.sin.data[t * half..(t + 1) * half],
            );
            for i in 0..options.heads {
                let (first, second) = out[i * d_k..(i + 1) * d_k].split_at_mut(half);
                let (a_row, b_row) = row[i * d_k..(i + 1) * d_k].split_at(half);
                for j in 0..half {
                    let (a, b, c, s) = (a_row[j], b_row[j], cos[j], sin[j]);
                    if inverse {
                        first[j] = a * c + b * s;
                        second[j] = b * c - a * s;
                    } else {
                        first[j] = a * c - b * s;
                        second[j] = a * s + b * c;
                    }
                }
            }
        }
        Array { data, shape: m.shape }
    }
}

fn require_shape(array: &Array, shape: Shape, name: &str, context: &str) -> Result<(), Error> {
    if array.shape != shape {
        return Err(Error::invalid(format!(
            "{context} requires {name} of shape {shape:?}, got {:?}",
            array.shape
        )));
    }
    Ok(())
}

/// `wq`, `wk`, `wv` `(h·d_k, d)` and `wo` `(d, h·d_k)`, `d` and `width` given.
fn require_weights(weights: [&Array; 4], d: usize, width: usize, context: &str) -> Result<(), Error> {
    let [wq, wk, wv, wo] = weights;
    for (w, name) in [(wq, "wq"), (wk, "wk"), (wv, "wv")] {
        require_shape(w, Shape::Matrix(width, d), name, context)?;
    }
    require_shape(wo, Shape::Matrix(d, width), "wo", context)
}

/// `(d, options)`, `d` the token width (`bo`'s length) and the projections' width `bq`'s: every
/// projection's weight and bias shaped to them.
fn shapes(
    projections: &Projections,
    heads: usize,
    causal: bool,
    dropout: f64,
    context: &str,
) -> Result<(usize, AttentionOptions), Error> {
    let (d, width) = (projections.bo.data.len(), projections.bq.data.len());
    if d == 0 {
        return Err(Error::invalid(format!("{context} requires a non-empty bo")));
    }
    let options = AttentionOptions::new(heads, width, causal, dropout, context)?;
    let weights = [projections.wq, projections.wk, projections.wv, projections.wo];
    require_weights(weights, d, width, context)?;
    for (b, name) in [(projections.bq, "bq"), (projections.bk, "bk"), (projections.bv, "bv")] {
        require_shape(b, Shape::Vector(width), name, context)?;
    }
    require_shape(projections.bo, Shape::Vector(d), "bo", context)?;
    Ok((d, options))
}

/// `(examples, tokens)` of `x`, 1D (one example) or 2D, `T * d` values per example.
fn examples_and_tokens(x: &Array, d: usize, name: &str, context: &str) -> Result<(usize, usize), Error> {
    let (rows, width) = match x.shape {
        Shape::Vector(n) => (1, n),
        Shape::Matrix(rows, n) => (rows, n),
    };
    if d == 0 || width == 0 || !width.is_multiple_of(d) {
        return Err(Error::invalid(format!(
            "{context} requires {name} of whole tokens of {d} features per example, got shape {:?}",
            x.shape
        )));
    }
    Ok((rows, width / d))
}

/// `x`'s rows of `d`: `(N * T, d)`, the same values.
fn token_rows(x: &Array, d: usize) -> Array {
    Array::from_matrix(x.data.clone(), x.data.len() / d, d)
}

fn columns(m: &Array) -> usize {
    match m.shape {
        Shape::Matrix(_, width) => width,
        Shape::Vector(width) => width,
    }
}

/// Example `n`'s head `i`: the `(T, width)` block of an `(N * T, h * width)` matrix at rows
/// `n·T..` and columns `i·width..`, copied; in one piece when the block spans whole rows.
fn head(m: &Array, n: usize, i: usize, tokens: usize, width: usize) -> Array {
    let stride = columns(m);
    let rows = &m.data[n * tokens * stride..(n + 1) * tokens * stride];
    let block = if width == stride {
        rows.to_vec()
    } else {
        let mut block = Vec::with_capacity(tokens * width);
        for row in rows.chunks_exact(stride) {
            block.extend_from_slice(&row[i * width..(i + 1) * width]);
        }
        block
    };
    Array::from_matrix(block, tokens, width)
}

/// `head`'s inverse for one example: its heads' `(T, width)` blocks side by side, appended to
/// `out` row by row; in one piece when there is one block.
fn extend_side_by_side(out: &mut Vec<f64>, blocks: &[Array]) {
    if let [block] = blocks {
        out.extend_from_slice(&block.data);
        return;
    }
    let width = columns(&blocks[0]);
    for t in 0..blocks[0].data.len() / width {
        for block in blocks {
            out.extend_from_slice(&block.data[t * width..(t + 1) * width]);
        }
    }
}

/// A `(T, T)` score matrix with each `S_tj`, `j > t`, set to `-inf`.
fn masked_causally(mut scores: Array, tokens: usize) -> Array {
    for (t, row) in scores.data.chunks_exact_mut(tokens).enumerate() {
        row[t + 1..].fill(f64::NEG_INFINITY);
    }
    scores
}

/// A `(T, T)` score matrix with each `S_tj` set to `-inf` where key `j` is in another segment than
/// query `t`, or a pad's (segment 0), but never on the diagonal (indrajala-ml's segments workplan,
/// D4): `segments` is one example's `T` segment ids. Computed per head from the ids, never stored.
fn masked_by_segment(mut scores: Array, segments: &[f64], tokens: usize) -> Array {
    for (t, row) in scores.data.chunks_exact_mut(tokens).enumerate() {
        for (j, score) in row.iter_mut().enumerate() {
            if j != t && (segments[j] != segments[t] || segments[j] == 0.0) {
                *score = f64::NEG_INFINITY;
            }
        }
    }
    scores
}

/// `P~ = P * M / keep`, elementwise, in that grouping: numpy's `P * M / keep`.
fn dropped(p: &Array, mask: &Array, keep: f64) -> Array {
    Array {
        data: p.data.iter().zip(&mask.data).map(|(&pv, &mv)| pv * mv / keep).collect(),
        shape: p.shape,
    }
}

/// Each value divided by `s`.
fn divided(m: Array, s: f64) -> Array {
    Array {
        data: m.data.iter().map(|&v| v / s).collect(),
        shape: m.shape,
    }
}

// project, attend, combine: the forward pass's blocks

/// `Q`, `K`, `V` from the `(N * T, d)` input rows: each the product, then the bias, over all heads;
/// with a `rotation`, `Q` and `K` then rotated.
fn project(
    rows: &Array,
    projections: &Projections,
    rotation: Option<Rotation>,
    tokens: usize,
    options: &AttentionOptions,
) -> Result<(Array, Array, Array), Error> {
    let q = affine_forward_batch(projections.wq, rows, projections.bq)?;
    let k = affine_forward_batch(projections.wk, rows, projections.bk)?;
    let v = affine_forward_batch(projections.wv, rows, projections.bv)?;
    Ok(match rotation {
        Some(rotation) => (
            rotation.rotated(&q, tokens, options, false),
            rotation.rotated(&k, tokens, options, false),
            v,
        ),
        None => (q, k, v),
    })
}

/// Per example and head, the weights and the weighted sum: `(p, h, mask)`, `p` `(N * T, h·T)` and
/// `h` `(N * T, h·d_k)`, from the packed `(N * T, h·d_k)` `q`, `k`, `v`. With `rng`, each head's
/// weights are dropped by a mask drawn from it, returned packed as `p`; without, `mask` is `None`.
fn attend_forward(
    q: &Array,
    k: &Array,
    v: &Array,
    tokens: usize,
    options: &AttentionOptions,
    mut rng: Option<&mut Generator>,
    segments: Option<&Array>,
) -> Result<(Array, Array, Option<Array>), Error> {
    let (heads, d_k, width, s) = (options.heads, options.key_size, options.width(), options.scale());
    let rows = q.data.len() / width;
    let mut p = Vec::with_capacity(rows * heads * tokens);
    let mut h = Vec::with_capacity(rows * width);
    let mut mask = rng.as_ref().map(|_| Vec::with_capacity(rows * heads * tokens));
    for n in 0..rows / tokens {
        let (mut p_n, mut h_n) = (Vec::with_capacity(heads), Vec::with_capacity(heads));
        let mut mask_n = Vec::with_capacity(if mask.is_some() { heads } else { 0 });
        for i in 0..heads {
            let (q_ni, k_ni, v_ni) = (
                head(q, n, i, tokens, d_k),
                head(k, n, i, tokens, d_k),
                head(v, n, i, tokens, d_k),
            );
            let mut scores = divided(matmul(&q_ni, &k_ni.transpose())?, s);
            if options.causal {
                scores = masked_causally(scores, tokens);
            }
            if let Some(segments) = segments {
                scores = masked_by_segment(scores, &segments.data[n * tokens..(n + 1) * tokens], tokens);
            }
            let p_ni = array_softmax(&scores);
            match rng.as_deref_mut() {
                Some(rng) => {
                    let m_ni = Array::from_matrix(
                        rng.draw_bernoulli_mask(options.dropout, tokens * tokens),
                        tokens,
                        tokens,
                    );
                    h_n.push(matmul(&dropped(&p_ni, &m_ni, options.keep()), &v_ni)?);
                    mask_n.push(m_ni);
                }
                None => h_n.push(matmul(&p_ni, &v_ni)?),
            }
            p_n.push(p_ni);
        }
        extend_side_by_side(&mut p, &p_n);
        extend_side_by_side(&mut h, &h_n);
        if let Some(mask) = mask.as_mut() {
            extend_side_by_side(mask, &mask_n);
        }
    }
    Ok((
        Array::from_matrix(p, rows, heads * tokens),
        Array::from_matrix(h, rows, width),
        mask.map(|mask| Array::from_matrix(mask, rows, heads * tokens)),
    ))
}

/// `H`, the heads side by side, through the output projection: one product over all heads.
fn combine(h: &Array, projections: &Projections) -> Result<Array, Error> {
    affine_forward_batch(projections.wo, h, projections.bo)
}

// the backward pass's blocks, last first

/// `dH = delta Wo`, `(N * T, h·d_k)`, from the `(N * T, d)` delta rows.
fn combine_backward(delta: &Array, wo: &Array) -> Result<Array, Error> {
    layer_downstream_batch(wo, delta)
}

/// Per example and head, the softmax's and the products' gradients: `(dq, dk, dv)`, each
/// `(N * T, h·d_k)`, from the forward pass's packed caches and `dh`, and its mask if it drew one.
fn attend_backward(
    caches: [&Array; 4],
    mask: Option<&Array>,
    dh: &Array,
    tokens: usize,
    options: &AttentionOptions,
) -> Result<(Array, Array, Array), Error> {
    let [q, k, v, p] = caches;
    let (heads, d_k, width, s) = (options.heads, options.key_size, options.width(), options.scale());
    let rows = dh.data.len() / width;
    let mut dq = Vec::with_capacity(rows * width);
    let mut dk = Vec::with_capacity(rows * width);
    let mut dv = Vec::with_capacity(rows * width);
    for n in 0..rows / tokens {
        let mut dq_n = Vec::with_capacity(heads);
        let mut dk_n = Vec::with_capacity(heads);
        let mut dv_n = Vec::with_capacity(heads);
        for i in 0..heads {
            let (q_ni, k_ni, v_ni) = (
                head(q, n, i, tokens, d_k),
                head(k, n, i, tokens, d_k),
                head(v, n, i, tokens, d_k),
            );
            let (p_ni, dh_ni) = (head(p, n, i, tokens, tokens), head(dh, n, i, tokens, d_k));
            let mut dp = matmul(&dh_ni, &v_ni.transpose())?;
            match mask {
                Some(mask) => {
                    let (m_ni, keep) = (head(mask, n, i, tokens, tokens), options.keep());
                    dv_n.push(matmul(&dropped(&p_ni, &m_ni, keep).transpose(), &dh_ni)?);
                    dp = dropped(&dp, &m_ni, keep);
                }
                None => dv_n.push(matmul(&p_ni.transpose(), &dh_ni)?),
            }
            let mut ds = Vec::with_capacity(tokens * tokens);
            for (dp_row, p_row) in dp.data.chunks_exact(tokens).zip(p_ni.data.chunks_exact(tokens)) {
                let r = dp_row.iter().zip(p_row).fold(0.0, |acc, (&dpv, &pv)| acc + dpv * pv);
                ds.extend(dp_row.iter().zip(p_row).map(|(&dpv, &pv)| pv * (dpv - r)));
            }
            let ds = Array::from_matrix(ds, tokens, tokens);
            dq_n.push(divided(matmul(&ds, &k_ni)?, s));
            dk_n.push(divided(matmul(&ds.transpose(), &q_ni)?, s));
        }
        extend_side_by_side(&mut dq, &dq_n);
        extend_side_by_side(&mut dk, &dk_n);
        extend_side_by_side(&mut dv, &dv_n);
    }
    Ok((
        Array::from_matrix(dq, rows, width),
        Array::from_matrix(dk, rows, width),
        Array::from_matrix(dv, rows, width),
    ))
}

/// `dX = (dQ Wq + dK Wk) + dV Wv`, `(N * T, d)`: three products over all heads, summed in this order.
fn project_backward(gradients: [&Array; 3], weights: [&Array; 3]) -> Result<Array, Error> {
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
    Ok(Array {
        data: dx,
        shape: dx_q.shape,
    })
}

/// The forward pass: `(a, q, k, v, p, h, mask)`, `a` in `x`'s shape, the caches as `(N * T, h·d_k)`
/// rows, `p` as `(N * T, h·T)` and `mask`, when the pass dropped, as `p`.
pub type ForwardResult = (Array, Array, Array, Array, Array, Array, Option<Array>);

/// How a forward pass drops its weights: `dropout`, and in `training` the generator it draws from.
struct Dropping<'a, 'py> {
    dropout: f64,
    training: bool,
    rng: Option<&'py mut Generator>,
    context: &'a str,
}

impl Dropping<'_, '_> {
    /// The generator to draw from: `rng` in a training pass with `dropout > 0`, which needs one;
    /// else `None`, and nothing is drawn.
    fn generator(&mut self) -> Result<Option<&mut Generator>, Error> {
        if !(self.training && self.dropout > 0.0) {
            return Ok(None);
        }
        match self.rng.as_deref_mut() {
            Some(rng) => Ok(Some(rng)),
            None => Err(Error::invalid(format!(
                "{} requires an rng to drop out in training, got dropout={} and none",
                self.context, self.dropout
            ))),
        }
    }
}

fn forward(
    x: &Array,
    projections: &Projections,
    heads: usize,
    causal: bool,
    mut dropping: Dropping,
    table: (Option<&Array>, Option<&Array>),
    segments: Option<&Array>,
) -> Result<ForwardResult, Error> {
    let context = dropping.context;
    let (d, options) = shapes(projections, heads, causal, dropping.dropout, context)?;
    let (examples, tokens) = examples_and_tokens(x, d, "x", context)?;
    if segments.is_some_and(|segments| segments.data.len() != examples * tokens) {
        let got = segments.map_or(0, |segments| segments.data.len());
        return Err(Error::invalid(format!(
            "{context} requires a segment id per token, {} ({examples} examples of {tokens}); got {}",
            examples * tokens,
            got
        )));
    }
    let rotation = Rotation::new(table.0, table.1, tokens, &options, context)?;
    let (q, k, v) = project(&token_rows(x, d), projections, rotation, tokens, &options)?;
    let (p, h, mask) = attend_forward(&q, &k, &v, tokens, &options, dropping.generator()?, segments)?;
    let out = combine(&h, projections)?;
    let a = Array {
        data: out.data,
        shape: x.shape,
    };
    Ok((a, q, k, v, p, h, mask))
}

/// `AttentionArrayLayer.forward`: one example, `x` 1D of `T * d` values, `d` `bo`'s length, in
/// `heads` heads across `bq`'s length, masked if `causal`, its weights dropped from `rng` in
/// `training` at `dropout > 0`, its queries and keys rotated by `cos` and `sin` when given, and
/// with `segments` (its `T` segment ids as floats) a key in another segment than its query, or a
/// pad's (0), masked but for the diagonal (indrajala-ml's segments workplan, D4).
/// Returns `(a, q, k, v, p, h, mask)`: `a` 1D, and what the backward
/// pass reads, `q`, `k`, `v` and `h` `(T, h·d_k)`, `p` `(T, h·T)` and `mask` as `p`, or `None`.
#[allow(clippy::too_many_arguments)]
pub fn attention_forward(
    x: &Array,
    wq: &Array,
    bq: &Array,
    wk: &Array,
    bk: &Array,
    wv: &Array,
    bv: &Array,
    wo: &Array,
    bo: &Array,
    heads: usize,
    causal: bool,
    dropout: f64,
    training: bool,
    rng: Option<&mut Generator>,
    cos: Option<&Array>,
    sin: Option<&Array>,
    segments: Option<&Array>,
) -> Result<ForwardResult, Error> {
    let context = "attention_forward";
    if !matches!(x.shape, Shape::Vector(_)) {
        return Err(Error::invalid(format!(
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
    let dropping = Dropping {
        dropout,
        training,
        rng,
        context,
    };
    forward(x, &projections, heads, causal, dropping, (cos, sin), segments)
}

/// `AttentionArrayLayer.forward_batch`: `x` 2D (`batch, T * d`), each example
/// `attention_forward`'s bits, masked if `causal` and by `segments` (`N * T` ids, examples then
/// tokens), dropped and rotated as it. Returns `(a, q, k, v, p, h,
/// mask)`, the caches over the batch's `(N * T)` rows, examples then tokens; the mask's draws are
/// the examples' in turn, so one example's are its batch of one's.
#[allow(clippy::too_many_arguments)]
pub fn attention_forward_batch(
    x: &Array,
    wq: &Array,
    bq: &Array,
    wk: &Array,
    bk: &Array,
    wv: &Array,
    bv: &Array,
    wo: &Array,
    bo: &Array,
    heads: usize,
    causal: bool,
    dropout: f64,
    training: bool,
    rng: Option<&mut Generator>,
    cos: Option<&Array>,
    sin: Option<&Array>,
    segments: Option<&Array>,
) -> Result<ForwardResult, Error> {
    let context = "attention_forward_batch";
    if !matches!(x.shape, Shape::Matrix(_, _)) {
        return Err(Error::invalid(format!(
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
    let dropping = Dropping {
        dropout,
        training,
        rng,
        context,
    };
    forward(x, &projections, heads, causal, dropping, (cos, sin), segments)
}

/// The cache `cache` (`(c, h·d_k)`, oldest first) with `row` (`(1, h·d_k)`) appended, the oldest
/// rows dropped past `window`.
fn appended(cache: &Array, row: &Array, window: usize, width: usize) -> Array {
    let mut data = Vec::with_capacity(cache.data.len() + width);
    data.extend_from_slice(&cache.data);
    data.extend_from_slice(&row.data);
    let rows = (data.len() / width).min(window);
    let dropped = data.len() / width - rows;
    Array::from_matrix(data.split_off(dropped * width), rows, width)
}

/// `AttentionArrayLayer.step` (indrajala-ml's generation workplan, D2, D3, D8): one token's decode
/// on a causal layer's cache. `x` 1D, the token's `d` features; `k_cache` and `v_cache` `(c,
/// h·d_k)`, the rotated keys and the values of the `c` tokens before it, oldest first; `cos` and
/// `sin`, when given, `(1, d_k / 2)`, its position's row of the rotary table. Project's blocks on
/// the one row give its `q`, `k` and `v`, `q` and `k` rotated; `k` and `v` are appended to the
/// caches, the oldest dropped past `window` (the window slides); each head attends from `q` to
/// every cached key, unmasked, since none is later, and combine projects the heads. Returns `(a,
/// k_cache, v_cache)`, `a` 1D and the updated caches.
///
/// Each block is the forward pass's: the projections a dense layer's per row, a score one chain
/// over `d_k` and a weighted sum one chain over the keys in order, so a step's bits are the full
/// pass's at that token, whose masked keys add exact zeros after (or, past the window, before) the
/// ones it reads.
#[allow(clippy::too_many_arguments)]
pub fn attention_step(
    x: &Array,
    wq: &Array,
    bq: &Array,
    wk: &Array,
    bk: &Array,
    wv: &Array,
    bv: &Array,
    wo: &Array,
    bo: &Array,
    k_cache: &Array,
    v_cache: &Array,
    heads: usize,
    window: usize,
    cos: Option<&Array>,
    sin: Option<&Array>,
) -> Result<(Array, Array, Array), Error> {
    let context = "attention_step";
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
    let (d, options) = shapes(&projections, heads, true, 0.0, context)?;
    require_shape(x, Shape::Vector(d), "x", context)?;
    if window == 0 {
        return Err(Error::invalid(format!(
            "{context} requires a window of at least one token"
        )));
    }
    let width = options.width();
    let cached = match k_cache.shape {
        Shape::Matrix(rows, w) if w == width => rows,
        shape => {
            return Err(Error::invalid(format!(
                "{context} requires k_cache of shape (c, {width}), got {shape:?}"
            )));
        }
    };
    require_shape(v_cache, Shape::Matrix(cached, width), "v_cache", context)?;
    let rotation = Rotation::new(cos, sin, 1, &options, context)?;
    let (q, k, v) = project(&token_rows(x, d), &projections, rotation, 1, &options)?;
    let (k_cache, v_cache) = (
        appended(k_cache, &k, window, width),
        appended(v_cache, &v, window, width),
    );
    let keys = k_cache.data.len() / width;
    let (d_k, s) = (options.key_size, options.scale());
    let mut h_i = Vec::with_capacity(options.heads);
    for i in 0..options.heads {
        let (q_i, k_i, v_i) = (
            head(&q, 0, i, 1, d_k),
            head(&k_cache, 0, i, keys, d_k),
            head(&v_cache, 0, i, keys, d_k),
        );
        let p_i = array_softmax(&divided(matmul(&q_i, &k_i.transpose())?, s));
        h_i.push(matmul(&p_i, &v_i)?);
    }
    let mut h = Vec::with_capacity(width);
    extend_side_by_side(&mut h, &h_i);
    let out = combine(&Array::from_matrix(h, 1, width), &projections)?;
    Ok((Array::from_vector(out.data), k_cache, v_cache))
}

/// `AttentionArrayLayer._backward`: from `delta_batch` (`dl/dout`, 2D `batch, T * d`, or 1D for
/// one example, whose caches are `attention_forward`'s) and the forward pass's `q`, `k`, `v` and
/// `p`, in `heads` heads across `wq`'s rows, returns `(dx, dq, dk, dv)`: `dx`, the downstream, in
/// `delta_batch`'s shape, and `dq`, `dk`, `dv` as `(N * T, h·d_k)` rows, which the gradients read.
/// A pass that dropped takes its `mask` (the forward op's, shaped as `p`) and its `dropout`; one
/// that rotated takes its `cos` and `sin`, and returns `dq` and `dk` rotated back.
#[allow(clippy::too_many_arguments)]
pub fn attention_downstream_batch(
    delta_batch: &Array,
    wq: &Array,
    wk: &Array,
    wv: &Array,
    wo: &Array,
    q: &Array,
    k: &Array,
    v: &Array,
    p: &Array,
    heads: usize,
    mask: Option<&Array>,
    dropout: f64,
    cos: Option<&Array>,
    sin: Option<&Array>,
) -> Result<(Array, Array, Array, Array), Error> {
    let context = "attention_downstream_batch";
    let (width, d) = match wq.shape {
        Shape::Matrix(width, d) => (width, d),
        Shape::Vector(d) => (d, 1),
    };
    // causal or not, the same: the mask is in `p`
    let options = AttentionOptions::new(heads, width, false, dropout, context)?;
    require_weights([wq, wk, wv, wo], d, width, context)?;
    let (examples, tokens) = examples_and_tokens(delta_batch, d, "delta_batch", context)?;
    let rows = examples * tokens;
    for (cache, name) in [(q, "q"), (k, "k"), (v, "v")] {
        require_shape(cache, Shape::Matrix(rows, width), name, context)?;
    }
    require_shape(p, Shape::Matrix(rows, heads * tokens), "p", context)?;
    if let Some(mask) = mask {
        require_shape(mask, p.shape, "mask", context)?;
    }
    let rotation = Rotation::new(cos, sin, tokens, &options, context)?;

    let dh = combine_backward(&token_rows(delta_batch, d), wo)?;
    let (dq, dk, dv) = attend_backward([q, k, v, p], mask, &dh, tokens, &options)?;
    // project's backward starts by rotating back: the gradients of the unrotated Q and K
    let (dq, dk) = match rotation {
        Some(rotation) => (
            rotation.rotated(&dq, tokens, &options, true),
            rotation.rotated(&dk, tokens, &options, true),
        ),
        None => (dq, dk),
    };
    let dx = project_backward([&dq, &dk, &dv], [wq, wk, wv])?;
    let dx = Array {
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
pub fn attention_accumulate_gradient_batch(
    delta_batch: &Array,
    x: &Array,
    h: &Array,
    dq: &Array,
    dk: &Array,
    dv: &Array,
    grad_wq: &Array,
    grad_bq: &Array,
    grad_wk: &Array,
    grad_bk: &Array,
    grad_wv: &Array,
    grad_bv: &Array,
    grad_wo: &Array,
    grad_bo: &Array,
) -> Result<(Array, Array, Array, Array, Array, Array, Array, Array), Error> {
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
