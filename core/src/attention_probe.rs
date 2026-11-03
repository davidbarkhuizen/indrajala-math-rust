//! Multi-head self-attention (indrajala-ml's multi-head attention workplan, D4; its context-length
//! workplan, D3, D12): one Rust function per pass of `AttentionArrayLayer` in
//! `indrajala_ml/model/layers/numpy/attention_array_layer.py`, each one call, so a pass crosses the
//! Python/Rust boundary once. Each pass validates the shapes and composes the crate-internal blocks,
//! project, attend and combine, in numpy's expressions and their grouping, with `h` heads of `d_k`
//! features and `[i]` head `i`'s block:
//!
//! ```text
//! project   Q = X Wq^T + bq;  K = X Wk^T + bk;  V = X Wv^T + bv     (T, h·d_k); K, V (T, g·d_k)
//! attend    per key block of b, in order, per example the rows reaching it, per head and row:
//!           S = (Q[i] K[i]^T) / s, masked -inf;  m' = max(m, max S);  e = exp(S - m')
//!           l = l exp(m - m') + fold(e);  H = H exp(m - m') + e V[i];  m = m';  at the end H / l
//! combine   H = [H[0] ... H[h-1]];  out = H Wo^T + bo
//! backward  dH = delta Wo;  D = fold(dH[i] H[i]) over d_k
//!           per key block: P = exp(S - m) / l;  dP = dH[i] V[i]^T;  dV[i] = P^T dH[i]
//!           dS = P (dP - D);  dQ[i] += dS K[i];  dK[i] = dS^T Q[i];  then dQ, dK / s
//!           dX = (dQ Wq + dK Wk) + dV Wv
//! ```
//!
//! with `s = sqrt(d_k)` and `fold` a left fold from the first value. The forward keeps `m` and `l`,
//! `(N * T, heads)`, never the `(T, T)` weights: memory linear in `T`. A key block's bounds are the
//! columns that are multiples of `b` (`BLOCK`, 16), so a decode's cache, given its first column, is
//! cut where the full pass's keys are. A masked score is `-inf`: causal, key `u` after query `t`'s
//! column; and by segment, another segment's key or a pad's (segment 0), never the query's own
//! (indrajala-ml's segments workplan, D4). A block no query of an example reaches is skipped for
//! it; a row wholly masked within a computed block shifts by 0 in place of its `-inf` max, so its
//! `e` is 0 and its `l` and `H` unchanged. Every sum here is a sequential fold, so skipping is
//! exact: a skipped block's terms would add exact zeros.
//!
//! The products between activations are `@` (`matmul`) per example, head and tile, a transpose
//! copied first; `exp` is the crate's correctly rounded `cr_exp`; project, combine and the
//! parameter gradients are a dense layer's on the `(N * T, d)` and `(N * T, h·d_k)` (`g·d_k`) rows
//! (`affine_forward_batch`, `layer_downstream_batch`, `layer_accumulate_gradient_batch`), one
//! product over all heads.
//!
//! Dropout (indrajala-ml's attention-dropout workplan, D3-D5; its context-length workplan, D12): a
//! training pass with `dropout > 0` draws, per key block, a 0/1 mask for every example, head, query
//! and key in that order from the network's `Generator`, numpy's `(N, h, T, b)` draw, so a pass
//! draws `N h T T` values, the masked ones included; `M = u >= dropout`, and `e` is dropped as
//! `e M / keep` before `e V`. The forward ops return a copy of the generator in its state before
//! the draws, from a copy of which the backward op draws the same masks again: `dV = (P kept)^T dH`
//! and `dP = dP kept`, `kept = M / keep`. Without `training`, and at `dropout = 0`, nothing is
//! drawn.
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
//! `K`. `V` isn't rotated.
//!
//! Grouped-query attention (indrajala-ml's grouped-query attention workplan, D2-D4): with
//! `key_value_heads` `g` dividing `h`, `K` and `V` hold `g` heads, and query head `i` reads
//! key/value head `i / (h / g)` in place, no copy; within a key block a group's heads run in
//! order, its tiles cut once. The backward pass adds a group's heads' `dK` and `dV` per key block,
//! the first head's assigned and each later one's added (a left fold in head order, numpy's
//! `fold`), and divides `dK` by the scale once, after. `g = h`, the default, is plain multi-head
//! attention, the same operations as before.
//!
//! The caches are packed: `Q`, `H` and their gradients `(N * T, h·d_k)`, head `i` in columns
//! `i·d_k..`; `K`, `V` and theirs `(N * T, g·d_k)`; and `m`, `l` `(N * T, heads)`.
//!
//! Cross-attention (indrajala-ml's cross-attention workplan, D3, D11, D13; numpy's
//! `CrossAttentionArrayLayer`): the `cross_attention_*` ops take the queries from the decoder's `T`
//! tokens of `d` features and the keys and values from the encoder's output, a memory of `S` tokens
//! of `d_m` features: `Q = X Wq^T + bq`, `K = M Wk^T + bk`, `V = M Wv^T + bv`, `Wk` and `Wv` `(g·d_k,
//! d_m)`. Attend and combine are self-attention's blocks over `T` queries and `S` keys, with the same
//! dropout draws per key block. Never causal, never rotary. The mask is by the queries' and the
//! keys' segments alone, with no own-column exception (`Masking::own_column`), so a pad's query may
//! reach no key: its row keeps `l = 0` and `H = 0`, and is divided by 1, so its output is `bo`. The
//! backward reads such a row's `m` as 0 and its `l` as 1, so its `P` is 0. Every self-attention row
//! has its own key, so these guards change none of its bits. The backward gives `dX = dQ Wq` and the
//! memory's gradient `dM = dK Wk + dV Wv`. The caches `K`, `V` and their gradients are `(N * S, g·d_k)`.

use crate::array::{Array, Shape};
use crate::cr_exp::{exp, exp_in_place};
use crate::fused::{affine_forward_batch, layer_accumulate_gradient_batch, layer_downstream_batch};
use crate::generator::Generator;
use crate::linalg::matmul;
use crate::Error;

/// What attend reads besides `Q`, `K` and `V`: the query and key/value head counts, each head's
/// width and the drop probability of each weight in training (the masks are each pass's
/// `Masking`).
#[derive(Clone, Copy)]
struct AttentionOptions {
    heads: usize,
    key_value_heads: usize,
    key_size: usize,
    dropout: f64,
}

impl AttentionOptions {
    /// `heads` heads across the query projection's `width = h·d_k` columns, sharing
    /// `key_value_heads` key/value heads, which divide them; `dropout` in `[0, 1)`.
    fn new(heads: usize, key_value_heads: usize, width: usize, dropout: f64, context: &str) -> Result<Self, Error> {
        if heads == 0 || width == 0 || !width.is_multiple_of(heads) {
            return Err(Error::invalid(format!(
                "{context} requires heads >= 1 dividing the projections' width {width}, got heads={heads}"
            )));
        }
        if key_value_heads == 0 || !heads.is_multiple_of(key_value_heads) {
            return Err(Error::invalid(format!(
                "{context} requires key_value_heads >= 1 dividing heads={heads}, got key_value_heads={key_value_heads}"
            )));
        }
        if !(0.0..1.0).contains(&dropout) {
            return Err(Error::invalid(format!(
                "{context} requires a dropout in [0, 1), got {dropout}"
            )));
        }
        Ok(AttentionOptions {
            heads,
            key_value_heads,
            key_size: width / heads,
            dropout,
        })
    }

    /// `g·d_k`, the key and value projections' width.
    fn kv_width(&self) -> usize {
        self.key_value_heads * self.key_size
    }

    /// Query head `i`'s key/value head, `i / (h / g)`.
    fn group(&self, i: usize) -> usize {
        i / (self.heads / self.key_value_heads)
    }

    /// Whether query head `i` is its group's first, which assigns the group's gradients.
    fn first_in_group(&self, i: usize) -> bool {
        i.is_multiple_of(self.heads / self.key_value_heads)
    }

    /// `1 - dropout`, each kept weight's divisor.
    fn keep(&self) -> f64 {
        1.0 - self.dropout
    }

    /// `h·d_k`, the query and output projections' width.
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

    /// The packed `(N * T, n·d_k)` rows `m`, `n` query or key/value heads, each head's features `j`
    /// and `j + d_k / 2` rotated by the row's token's angles: `(a c - b s, a s + b c)`; or back,
    /// `inverse`, by the transpose: `(a c + b s, b c - a s)`. Each entry two products and a sum, as
    /// numpy's elementwise expressions, so the same bits.
    fn rotated(&self, m: &Array, tokens: usize, options: &AttentionOptions, inverse: bool) -> Array {
        let (d_k, width) = (options.key_size, columns(m));
        let half = d_k / 2;
        let mut data = m.data.clone();
        for (r, (row, out)) in m.data.chunks_exact(width).zip(data.chunks_exact_mut(width)).enumerate() {
            let t = r % tokens;
            let (cos, sin) = (
                &self.cos.data[t * half..(t + 1) * half],
                &self.sin.data[t * half..(t + 1) * half],
            );
            for i in 0..width / d_k {
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

/// `wq` `(h·d_k, d)`, `wk`, `wv` `(g·d_k, d_m)` and `wo` `(d, h·d_k)`, `(d, d_m)` and the options
/// given: `d_m` the keys' and values' input width, `d` in self-attention.
fn require_weights(
    weights: [&Array; 4],
    (d, d_m): (usize, usize),
    options: &AttentionOptions,
    context: &str,
) -> Result<(), Error> {
    let [wq, wk, wv, wo] = weights;
    let (width, kv_width) = (options.width(), options.kv_width());
    require_shape(wq, Shape::Matrix(width, d), "wq", context)?;
    for (w, name) in [(wk, "wk"), (wv, "wv")] {
        require_shape(w, Shape::Matrix(kv_width, d_m), name, context)?;
    }
    require_shape(wo, Shape::Matrix(d, width), "wo", context)
}

/// `(d, options)`, `d` the token width (`bo`'s length) and the projections' width `bq`'s: every
/// projection's weight and bias shaped to them, `wk` and `wv` reading `memory_width` features (`d`
/// when `None`, self-attention).
fn shapes(
    projections: &Projections,
    heads: (usize, usize),
    dropout: f64,
    memory_width: Option<usize>,
    context: &str,
) -> Result<(usize, AttentionOptions), Error> {
    let (d, width) = (projections.bo.data.len(), projections.bq.data.len());
    if d == 0 {
        return Err(Error::invalid(format!("{context} requires a non-empty bo")));
    }
    let options = AttentionOptions::new(heads.0, heads.1, width, dropout, context)?;
    let weights = [projections.wq, projections.wk, projections.wv, projections.wo];
    require_weights(weights, (d, memory_width.unwrap_or(d)), &options, context)?;
    require_shape(projections.bq, Shape::Vector(width), "bq", context)?;
    for (b, name) in [(projections.bk, "bk"), (projections.bv, "bv")] {
        require_shape(b, Shape::Vector(options.kv_width()), name, context)?;
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

/// Each value divided by `s`.
fn divided(m: Array, s: f64) -> Array {
    Array {
        data: m.data.iter().map(|&v| v / s).collect(),
        shape: m.shape,
    }
}

/// `BLOCK`, the key blocks' default size (indrajala-ml's context-length workplan, D3: 16, measured).
pub const BLOCK: usize = 16;

/// Which keys one example's queries may attend to (indrajala-ml's context-length workplan, D3):
/// causal, key `u` at most query `t`'s column `t + offset` (a decode's one query is the keys' last,
/// `offset = keys - 1`); and with segment ids, the queries' and the keys', another segment's key or
/// a pad's (0) masked, but never the query's own column when `own_column` (indrajala-ml's segments
/// workplan, D4). Cross-attention's queries and keys are two sequences (its cross-attention
/// workplan, D3): `own_column` false masks by segment alone, so a pad's query may reach no key.
#[derive(Clone, Copy)]
struct Masking<'a> {
    causal: bool,
    offset: usize,
    segments: Option<(&'a [f64], &'a [f64])>,
    own_column: bool,
}

impl Masking<'_> {
    fn allowed(&self, t: usize, u: usize) -> bool {
        let column = t + self.offset;
        if self.causal && u > column {
            return false;
        }
        match self.segments {
            Some((queries, keys)) => (self.own_column && u == column) || (keys[u] == queries[t] && keys[u] != 0.0),
            None => true,
        }
    }

    /// Whether any score of rows `r0..r1` against keys `k0..k1` is masked.
    fn masks(&self, r0: usize, r1: usize, k0: usize, k1: usize) -> bool {
        (self.causal && k1 - 1 > r0 + self.offset)
            || self.segments.is_some() && (r0..r1).any(|t| (k0..k1).any(|u| !self.allowed(t, u)))
    }

    /// The rows `r0..r1` of `queries` reaching keys `k0..k1`, the smallest range holding every
    /// query that may attend to one of them; `None` when no query may (the block is skipped).
    fn rows(&self, queries: usize, k0: usize, k1: usize) -> Option<(usize, usize)> {
        let r0 = if self.causal { k0.saturating_sub(self.offset) } else { 0 };
        if r0 >= queries {
            return None;
        }
        if self.segments.is_none() {
            return Some((r0, queries));
        }
        let reaches = |t: usize| (k0..k1).any(|u| self.allowed(t, u));
        let first = (r0..queries).find(|&t| reaches(t))?;
        let last = (first..queries).rev().find(|&t| reaches(t))?;
        Some((first, last + 1))
    }
}

/// The key blocks `k0..k1` of `keys` keys whose first is at column `origin`, bounded at the columns
/// that are multiples of `block`: a decode's cache is cut where the full pass over every column is.
fn key_blocks(keys: usize, block: usize, origin: usize) -> Vec<(usize, usize)> {
    let first = match (block - origin % block) % block {
        0 => block,
        first => first,
    };
    let mut bounds = vec![0];
    bounds.extend((first..keys).step_by(block));
    bounds.push(keys);
    bounds
        .windows(2)
        .filter(|w| w[0] < w[1])
        .map(|w| (w[0], w[1]))
        .collect()
}

/// Rows `r0..r1` and columns `c0..c0 + width` of a packed matrix, copied.
fn tile(m: &Array, r0: usize, r1: usize, c0: usize, width: usize) -> Array {
    let stride = columns(m);
    let mut data = Vec::with_capacity((r1 - r0) * width);
    for r in r0..r1 {
        data.extend_from_slice(&m.data[r * stride + c0..r * stride + c0 + width]);
    }
    Array::from_matrix(data, r1 - r0, width)
}

/// A left fold from the first value, as numpy's `cumsum` along a row: `((x0 + x1) + x2) + ...`.
fn fold(values: &[f64]) -> f64 {
    values[1..].iter().fold(values[0], |acc, &x| acc + x)
}

/// The shapes a tiled pass runs over: `examples` examples of `queries` queries and `keys` keys.
#[derive(Clone, Copy)]
struct Extent {
    examples: usize,
    queries: usize,
    keys: usize,
    block: usize,
    origin: usize,
}

/// Example `n`'s key/value head `group`'s tile of keys or values `k0..k1`, `(k1 - k0, d_k)`.
fn group_tile(m: &Array, n: usize, group: usize, k0: usize, k1: usize, extent: Extent, d_k: usize) -> Array {
    tile(m, n * extent.keys + k0, n * extent.keys + k1, group * d_k, d_k)
}

/// One key block's scores against rows `r0..r1` of example `n`'s head `i`, `k_t` its key/value
/// head's keys `k0..k1` transposed: `(Q K^T) / s`, masked scores `-inf`.
fn scores(
    q: &Array,
    k_t: &Array,
    at: (usize, usize, usize, usize, usize, usize),
    extent: Extent,
    options: &AttentionOptions,
    masking: &Masking,
) -> Result<Array, Error> {
    let (n, i, r0, r1, k0, k1) = at;
    let d_k = options.key_size;
    let q_t = tile(q, n * extent.queries + r0, n * extent.queries + r1, i * d_k, d_k);
    let mut s = divided(matmul(&q_t, k_t)?, options.scale());
    if masking.masks(r0, r1, k0, k1) {
        let b = k1 - k0;
        for (r, row) in s.data.chunks_exact_mut(b).enumerate() {
            for (j, score) in row.iter_mut().enumerate() {
                if !masking.allowed(r0 + r, k0 + j) {
                    *score = f64::NEG_INFINITY;
                }
            }
        }
    }
    Ok(s)
}

/// A key block's dropout mask over every example, head, query and key in that order, `None` when
/// the pass drops nothing (indrajala-ml's context-length workplan, D12).
fn block_mask(
    rng: &mut Option<&mut Generator>,
    extent: Extent,
    options: &AttentionOptions,
    b: usize,
) -> Option<Vec<f64>> {
    rng.as_deref_mut()
        .map(|rng| rng.draw_bernoulli_mask(options.dropout, extent.examples * options.heads * extent.queries * b))
}

/// The tiled forward (indrajala-ml's context-length workplan, D3, D12): `(h, m, l)`, `h` the packed
/// `(N * T_q, h·d_k)` output and `m` and `l` each query's running max and sum, `(N * T_q, heads)`,
/// from the packed `q` `(N * T_q, h·d_k)` and `k`, `v` `(N * T_k, g·d_k)`. Per key block, in
/// order, a mask drawn from `rng` for every example, head, query and key; then per example the
/// rows reaching it, per key/value head its tiles, and per query head of its group and row, `m' = max(m, the block's row max)`, `e = exp(S - m')` (0
/// in place of an all-masked row's `m'`), `l = l exp(m - m') + fold(e)`, `H = H exp(m - m') + e V`
/// (`e` dropped as `e M / keep`); at the end `H / l`. numpy's `tiled_forward`, expression for
/// expression.
fn tiled_forward<'s>(
    qkv: [&Array; 3],
    extent: Extent,
    options: &AttentionOptions,
    masking: impl Fn(usize) -> Masking<'s>,
    mut rng: Option<&mut Generator>,
) -> Result<(Array, Array, Array), Error> {
    let [q, k, v] = qkv;
    let (heads, d_k, width) = (options.heads, options.key_size, options.width());
    let rows = extent.examples * extent.queries;
    let mut m = vec![f64::NEG_INFINITY; rows * heads];
    let mut l = vec![0.0; rows * heads];
    let mut h = vec![0.0; rows * width];
    for (k0, k1) in key_blocks(extent.keys, extent.block, extent.origin) {
        let b = k1 - k0;
        let mask = block_mask(&mut rng, extent, options, b);
        for n in 0..extent.examples {
            let masking = masking(n);
            let Some((r0, r1)) = masking.rows(extent.queries, k0, k1) else {
                continue;
            };
            let (mut k_t, mut v_t) = (Array::from_vector(vec![]), Array::from_vector(vec![]));
            for i in 0..heads {
                if options.first_in_group(i) {
                    let group = options.group(i);
                    k_t = group_tile(k, n, group, k0, k1, extent, d_k).transpose();
                    v_t = group_tile(v, n, group, k0, k1, extent, d_k);
                }
                let s = scores(q, &k_t, (n, i, r0, r1, k0, k1), extent, options, &masking)?;
                let mut e = vec![0.0; (r1 - r0) * b];
                let mut alphas = Vec::with_capacity(r1 - r0);
                for (r, (row, e_row)) in s.data.chunks_exact(b).zip(e.chunks_exact_mut(b)).enumerate() {
                    let t = r0 + r;
                    let at = (n * extent.queries + t) * heads + i;
                    let m_old = m[at];
                    let m_new = row.iter().fold(m_old, |acc, &x| acc.max(x));
                    let shift = if m_new == f64::NEG_INFINITY { 0.0 } else { m_new };
                    for (x, &score) in e_row.iter_mut().zip(row) {
                        *x = score - shift;
                    }
                    exp_in_place(e_row);
                    let alpha = exp(m_old - shift);
                    l[at] = l[at] * alpha + fold(e_row);
                    m[at] = m_new;
                    alphas.push(alpha);
                    if let Some(mask) = &mask {
                        let start = ((n * heads + i) * extent.queries + t) * b;
                        for (x, &kept) in e_row.iter_mut().zip(&mask[start..start + b]) {
                            *x = *x * kept / options.keep();
                        }
                    }
                }
                let ev = matmul(&Array::from_matrix(e, r1 - r0, b), &v_t)?;
                for (r, ev_row) in ev.data.chunks_exact(d_k).enumerate() {
                    let start = (n * extent.queries + r0 + r) * width + i * d_k;
                    for (out, &x) in h[start..start + d_k].iter_mut().zip(ev_row) {
                        *out = *out * alphas[r] + x;
                    }
                }
            }
        }
    }
    for (r, row) in h.chunks_exact_mut(width).enumerate() {
        for i in 0..heads {
            // a row no key is allowed to (a cross-attention pad's query) keeps l = 0 and H = 0,
            // divided by 1; every self-attention row has its own key, so l > 0 there
            let sum = if l[r * heads + i] == 0.0 { 1.0 } else { l[r * heads + i] };
            for x in &mut row[i * d_k..(i + 1) * d_k] {
                *x /= sum;
            }
        }
    }
    Ok((
        Array::from_matrix(h, rows, width),
        Array::from_matrix(m, rows, heads),
        Array::from_matrix(l, rows, heads),
    ))
}

/// The tiled backward: `(dq, dk, dv)`, packed as `q`, `k`, `v` (`dk` and `dv` a group's heads'
/// added per key block in head order, the first's assigned), from `dh` and the forward's `q`,
/// `k`, `v`, `h`, `m` and `l`. Per key block, its masks drawn again from `rng` (a copy of the
/// forward's generator before its draws), `P = exp(S - m) / l` recomputed, `dP = dH V^T`, with
/// `kept = M / keep`: `dV = (P kept)^T dH` and `dP = dP kept`; `dS = P (dP - D)`, `D` the fold of
/// `dH H` over `d_k` (FlashAttention-2's); `dQ` the blocks' `dS K` summed in key order and `dK =
/// dS^T Q`; then `dQ` and `dK` divided by the scale. numpy's `tiled_backward`.
fn tiled_backward<'s>(
    dh: &Array,
    saved: [&Array; 6],
    extent: Extent,
    options: &AttentionOptions,
    masking: impl Fn(usize) -> Masking<'s>,
    mut rng: Option<&mut Generator>,
) -> Result<(Array, Array, Array), Error> {
    let [q, k, v, h, m, l] = saved;
    let (heads, d_k, width, s) = (options.heads, options.key_size, options.width(), options.scale());
    let kv_width = options.kv_width();
    let rows = extent.examples * extent.queries;
    let row_terms: Vec<f64> = dh
        .data
        .chunks_exact(d_k)
        .zip(h.data.chunks_exact(d_k))
        .map(|(dh_j, h_j)| {
            let products: Vec<f64> = dh_j.iter().zip(h_j).map(|(&a, &b)| a * b).collect();
            fold(&products)
        })
        .collect();
    let mut dq = vec![0.0; rows * width];
    let mut dk = vec![0.0; extent.examples * extent.keys * kv_width];
    let mut dv = vec![0.0; extent.examples * extent.keys * kv_width];
    for (k0, k1) in key_blocks(extent.keys, extent.block, extent.origin) {
        let b = k1 - k0;
        let mask = block_mask(&mut rng, extent, options, b);
        for n in 0..extent.examples {
            let masking = masking(n);
            let Some((r0, r1)) = masking.rows(extent.queries, k0, k1) else {
                continue;
            };
            let empty = || Array::from_vector(vec![]);
            let (mut k_t, mut k_tt, mut v_t) = (empty(), empty(), empty());
            for i in 0..heads {
                let (group, first) = (options.group(i), options.first_in_group(i));
                if first {
                    k_t = group_tile(k, n, group, k0, k1, extent, d_k);
                    k_tt = k_t.transpose();
                    v_t = group_tile(v, n, group, k0, k1, extent, d_k);
                }
                let mut p = scores(q, &k_tt, (n, i, r0, r1, k0, k1), extent, options, &masking)?;
                for (r, row) in p.data.chunks_exact_mut(b).enumerate() {
                    let at = (n * extent.queries + r0 + r) * heads + i;
                    // a row no key is allowed to (m = -inf, l = 0) reads m as 0 and l as 1, so its P
                    // is exp(-inf) / 1 = 0, as the forward divides its H; a no-op for every other row
                    let (m_at, l_at) = (m.data[at], l.data[at]);
                    let shift = if m_at == f64::NEG_INFINITY { 0.0 } else { m_at };
                    let sum = if l_at == 0.0 { 1.0 } else { l_at };
                    for x in row.iter_mut() {
                        *x -= shift;
                    }
                    exp_in_place(row);
                    for x in row.iter_mut() {
                        *x /= sum;
                    }
                }
                let dh_t = tile(dh, n * extent.queries + r0, n * extent.queries + r1, i * d_k, d_k);
                let mut dp = matmul(&dh_t, &v_t.transpose())?;
                let dv_t = match &mask {
                    Some(mask) => {
                        let mut kept = Vec::with_capacity((r1 - r0) * b);
                        for t in r0..r1 {
                            let start = ((n * heads + i) * extent.queries + t) * b;
                            kept.extend(mask[start..start + b].iter().map(|&x| x / options.keep()));
                        }
                        let dropped = Array::from_matrix(
                            p.data.iter().zip(&kept).map(|(&pv, &kv)| pv * kv).collect(),
                            r1 - r0,
                            b,
                        );
                        for (x, &kv) in dp.data.iter_mut().zip(&kept) {
                            *x *= kv;
                        }
                        matmul(&dropped.transpose(), &dh_t)?
                    }
                    None => matmul(&p.transpose(), &dh_t)?,
                };
                let mut ds = p.data;
                for (r, (ds_row, dp_row)) in ds.chunks_exact_mut(b).zip(dp.data.chunks_exact(b)).enumerate() {
                    let term = row_terms[(n * extent.queries + r0 + r) * heads + i];
                    for (x, &dpv) in ds_row.iter_mut().zip(dp_row) {
                        *x *= dpv - term;
                    }
                }
                let ds = Array::from_matrix(ds, r1 - r0, b);
                let q_t = tile(q, n * extent.queries + r0, n * extent.queries + r1, i * d_k, d_k);
                let dq_t = matmul(&ds, &k_t)?;
                for (r, row) in dq_t.data.chunks_exact(d_k).enumerate() {
                    let start = (n * extent.queries + r0 + r) * width + i * d_k;
                    for (out, &x) in dq[start..start + d_k].iter_mut().zip(row) {
                        *out += x;
                    }
                }
                let dk_t = matmul(&ds.transpose(), &q_t)?;
                for (j, (dk_row, dv_row)) in dk_t.data.chunks_exact(d_k).zip(dv_t.data.chunks_exact(d_k)).enumerate() {
                    let start = (n * extent.keys + k0 + j) * kv_width + group * d_k;
                    if first {
                        dk[start..start + d_k].copy_from_slice(dk_row);
                        dv[start..start + d_k].copy_from_slice(dv_row);
                    } else {
                        for (out, &x) in dk[start..start + d_k].iter_mut().zip(dk_row) {
                            *out += x;
                        }
                        for (out, &x) in dv[start..start + d_k].iter_mut().zip(dv_row) {
                            *out += x;
                        }
                    }
                }
            }
        }
    }
    let keys = extent.examples * extent.keys;
    Ok((
        divided(Array::from_matrix(dq, rows, width), s),
        divided(Array::from_matrix(dk, keys, kv_width), s),
        Array::from_matrix(dv, keys, kv_width),
    ))
}

// project, attend, combine: the forward pass's blocks

/// `Q` from the `(N * T, d)` input rows and `K`, `V` from `kv_rows`, the same rows in self-attention
/// and the memory's `(N * S, d_m)` in cross-attention: each the product, then the bias, over all
/// heads; with a `rotation`, `Q` and `K` then rotated.
fn project(
    rows: &Array,
    kv_rows: &Array,
    projections: &Projections,
    rotation: Option<Rotation>,
    tokens: usize,
    options: &AttentionOptions,
) -> Result<(Array, Array, Array), Error> {
    let q = affine_forward_batch(projections.wq, rows, projections.bq)?;
    let k = affine_forward_batch(projections.wk, kv_rows, projections.bk)?;
    let v = affine_forward_batch(projections.wv, kv_rows, projections.bv)?;
    Ok(match rotation {
        Some(rotation) => (
            rotation.rotated(&q, tokens, options, false),
            rotation.rotated(&k, tokens, options, false),
            v,
        ),
        None => (q, k, v),
    })
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

/// The forward pass: `(a, q, k, v, h, m, l, dropout_rng)`, `a` in `x`'s shape, `q`, `k`, `v` and
/// `h` as `(N * T, h·d_k)` rows (`k`, `v` `g·d_k`), `m` and `l` `(N * T, heads)`, and `dropout_rng`, when the pass
/// dropped, a copy of the generator in its state before the pass's draws, which the backward op
/// draws the same masks from (indrajala-ml's context-length workplan, D12).
pub type ForwardResult = (Array, Array, Array, Array, Array, Array, Array, Option<Generator>);

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

/// `block >= 1`, the key blocks' size.
fn require_block(block: usize, context: &str) -> Result<(), Error> {
    if block == 0 {
        return Err(Error::invalid(format!(
            "{context} requires key blocks of at least one key, got block=0"
        )));
    }
    Ok(())
}

/// `segments`, when given, one id per token of `examples` examples of `tokens`.
fn require_segments(segments: Option<&Array>, examples: usize, tokens: usize, context: &str) -> Result<(), Error> {
    match segments {
        Some(segments) if segments.data.len() != examples * tokens => Err(Error::invalid(format!(
            "{context} requires a segment id per token, {} ({examples} examples of {tokens}); got {}",
            examples * tokens,
            segments.data.len()
        ))),
        _ => Ok(()),
    }
}

/// Example `n`'s masking in a pass over `tokens` tokens: causal or not, by its segment ids if given.
fn example_masking<'a>(causal: bool, segments: Option<&'a Array>, tokens: usize) -> impl Fn(usize) -> Masking<'a> + 'a {
    move |n| Masking {
        causal,
        offset: 0,
        segments: segments.map(|s| {
            let ids = &s.data[n * tokens..(n + 1) * tokens];
            (ids, ids)
        }),
        own_column: true,
    }
}

/// Example `n`'s cross-attention masking, `queries` and `keys` tokens a side: never causal, by its
/// queries' and its keys' segment ids if given, without the own-column exception.
fn cross_masking<'a>(
    segments: Option<(&'a Array, &'a Array)>,
    queries: usize,
    keys: usize,
) -> impl Fn(usize) -> Masking<'a> + 'a {
    move |n| Masking {
        causal: false,
        offset: 0,
        segments: segments.map(|(q, k)| {
            (
                &q.data[n * queries..(n + 1) * queries],
                &k.data[n * keys..(n + 1) * keys],
            )
        }),
        own_column: false,
    }
}

#[allow(clippy::too_many_arguments)]
fn forward(
    x: &Array,
    projections: &Projections,
    heads: (usize, usize),
    causal: bool,
    mut dropping: Dropping,
    table: (Option<&Array>, Option<&Array>),
    segments: Option<&Array>,
    block: usize,
) -> Result<ForwardResult, Error> {
    let context = dropping.context;
    let (d, options) = shapes(projections, heads, dropping.dropout, None, context)?;
    require_block(block, context)?;
    let (examples, tokens) = examples_and_tokens(x, d, "x", context)?;
    require_segments(segments, examples, tokens, context)?;
    let rotation = Rotation::new(table.0, table.1, tokens, &options, context)?;
    let rows = token_rows(x, d);
    let (q, k, v) = project(&rows, &rows, projections, rotation, tokens, &options)?;
    // a copy in the state before the draws, for the backward op (D12); the tiles draw from the caller's
    let saved = dropping.generator()?.map(|rng| rng.clone());
    let extent = Extent {
        examples,
        queries: tokens,
        keys: tokens,
        block,
        origin: 0,
    };
    let masking = example_masking(causal, segments, tokens);
    let (h, m, l) = tiled_forward([&q, &k, &v], extent, &options, masking, dropping.generator()?)?;
    let out = combine(&h, projections)?;
    let a = Array {
        data: out.data,
        shape: x.shape,
    };
    Ok((a, q, k, v, h, m, l, saved))
}

/// `AttentionArrayLayer.forward`: one example, `x` 1D of `T * d` values, `d` `bo`'s length, in
/// `heads` heads across `bq`'s length, masked if `causal`, its weights dropped from `rng` in
/// `training` at `dropout > 0`, its queries and keys rotated by `cos` and `sin` when given, and
/// with `segments` (its `T` segment ids as floats) a key in another segment than its query, or a
/// pad's (0), masked but for the diagonal (indrajala-ml's segments workplan, D4); tiled in key
/// blocks of `block` (indrajala-ml's context-length workplan, D3). Returns `(a, q, k, v, h, m, l,
/// dropout_rng)`: `a` 1D, and what the backward pass reads, `q` and `h` `(T, h·d_k)`, `k`, `v` `(T, g·d_k)`, `m`
/// and `l` `(T, heads)`, and the generator's copy before the draws, or `None`.
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
    key_value_heads: usize,
    causal: bool,
    dropout: f64,
    training: bool,
    rng: Option<&mut Generator>,
    cos: Option<&Array>,
    sin: Option<&Array>,
    segments: Option<&Array>,
    block: usize,
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
    forward(
        x,
        &projections,
        (heads, key_value_heads),
        causal,
        dropping,
        (cos, sin),
        segments,
        block,
    )
}

/// `AttentionArrayLayer.forward_batch`: `x` 2D (`batch, T * d`), masked if `causal` and by
/// `segments` (`N * T` ids, examples then tokens), dropped and rotated as `attention_forward`,
/// tiled in key blocks of `block`. Returns `(a, q, k, v, h, m, l, dropout_rng)`, the caches over the
/// batch's `(N * T)` rows, examples then tokens. A key block's draws are every example's, head's,
/// query's and key's in that order (D12), numpy's `(N, h, T, b)` draw.
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
    key_value_heads: usize,
    causal: bool,
    dropout: f64,
    training: bool,
    rng: Option<&mut Generator>,
    cos: Option<&Array>,
    sin: Option<&Array>,
    segments: Option<&Array>,
    block: usize,
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
    forward(
        x,
        &projections,
        (heads, key_value_heads),
        causal,
        dropping,
        (cos, sin),
        segments,
        block,
    )
}

/// The cache `cache` (`(c, g·d_k)`, oldest first) with `row` (`(1, g·d_k)`) appended, the oldest
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
/// g·d_k)`, the rotated keys and the values of the `c` tokens before it, oldest first; `cos` and
/// `sin`, when given, `(1, d_k / 2)`, its position's row of the rotary table. Project's blocks on
/// the one row give its `q`, `k` and `v`, `q` and `k` rotated; `k` and `v` are appended to the
/// caches, the oldest dropped past `window` (the window slides); each head attends from `q` to
/// every cached key, unmasked, since none is later, through the forward pass's tiles, the cache cut
/// into key blocks of `block` at the columns that are multiples of it, `origin` the first cached
/// key's column (after the append); and combine projects the heads. Returns `(a, k_cache, v_cache)`,
/// `a` 1D and the updated caches.
///
/// Each block is the forward pass's: the projections a dense layer's per row, a score one chain
/// over `d_k`, each key block's online softmax and weighted sum the same expressions on the same
/// keys, so a step's bits are the full pass's at that token, whose masked keys and skipped blocks
/// change nothing.
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
    key_value_heads: usize,
    window: usize,
    cos: Option<&Array>,
    sin: Option<&Array>,
    origin: usize,
    block: usize,
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
    let (d, options) = shapes(&projections, (heads, key_value_heads), 0.0, None, context)?;
    require_shape(x, Shape::Vector(d), "x", context)?;
    require_block(block, context)?;
    if window == 0 {
        return Err(Error::invalid(format!(
            "{context} requires a window of at least one token"
        )));
    }
    let width = options.kv_width();
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
    let rows = token_rows(x, d);
    let (q, k, v) = project(&rows, &rows, &projections, rotation, 1, &options)?;
    let (k_cache, v_cache) = (
        appended(k_cache, &k, window, width),
        appended(v_cache, &v, window, width),
    );
    let keys = k_cache.data.len() / width;
    let extent = Extent {
        examples: 1,
        queries: 1,
        keys,
        block,
        origin,
    };
    let masking = |_| Masking {
        causal: true,
        offset: keys - 1,
        segments: None,
        own_column: true,
    };
    let (h, _m, _l) = tiled_forward([&q, &k_cache, &v_cache], extent, &options, masking, None)?;
    let out = combine(&h, &projections)?;
    Ok((Array::from_vector(out.data), k_cache, v_cache))
}

/// `AttentionArrayLayer._backward`: from `delta_batch` (`dl/dout`, 2D `batch, T * d`, or 1D for
/// one example, whose caches are `attention_forward`'s) and the forward pass's `q`, `k`, `v`, `h`,
/// `m` and `l`, in `heads` heads across `wq`'s rows, masked as the forward was (`causal`,
/// `segments`) and tiled in its key blocks of `block`, returns `(dx, dq, dk, dv)`: `dx`, the
/// downstream, in `delta_batch`'s shape, and `dq` `(N * T, h·d_k)` and `dk`, `dv` `(N * T, g·d_k)`, which the
/// gradients read. A pass that dropped takes its `dropout` and its `dropout_rng` (the forward op's
/// copy), from a copy of which it draws the same masks again; one that rotated takes its `cos` and
/// `sin`, and returns `dq` and `dk` rotated back.
#[allow(clippy::too_many_arguments)]
pub fn attention_downstream_batch(
    delta_batch: &Array,
    wq: &Array,
    wk: &Array,
    wv: &Array,
    wo: &Array,
    caches: [&Array; 6],
    heads: usize,
    key_value_heads: usize,
    causal: bool,
    segments: Option<&Array>,
    dropout: f64,
    dropout_rng: Option<&Generator>,
    cos: Option<&Array>,
    sin: Option<&Array>,
    block: usize,
) -> Result<(Array, Array, Array, Array), Error> {
    let context = "attention_downstream_batch";
    let [q, k, v, h, m, l] = caches;
    let (width, d) = match wq.shape {
        Shape::Matrix(width, d) => (width, d),
        Shape::Vector(d) => (d, 1),
    };
    let options = AttentionOptions::new(heads, key_value_heads, width, dropout, context)?;
    require_weights([wq, wk, wv, wo], (d, d), &options, context)?;
    require_block(block, context)?;
    let (examples, tokens) = examples_and_tokens(delta_batch, d, "delta_batch", context)?;
    let rows = examples * tokens;
    for (cache, name) in [(q, "q"), (h, "h")] {
        require_shape(cache, Shape::Matrix(rows, width), name, context)?;
    }
    for (cache, name) in [(k, "k"), (v, "v")] {
        require_shape(cache, Shape::Matrix(rows, options.kv_width()), name, context)?;
    }
    for (cache, name) in [(m, "m"), (l, "l")] {
        require_shape(cache, Shape::Matrix(rows, heads), name, context)?;
    }
    require_segments(segments, examples, tokens, context)?;
    let rotation = Rotation::new(cos, sin, tokens, &options, context)?;

    let dh = combine_backward(&token_rows(delta_batch, d), wo)?;
    let extent = Extent {
        examples,
        queries: tokens,
        keys: tokens,
        block,
        origin: 0,
    };
    let masking = example_masking(causal, segments, tokens);
    let mut rng = if dropout > 0.0 { dropout_rng.cloned() } else { None };
    let (dq, dk, dv) = tiled_backward(&dh, [q, k, v, h, m, l], extent, &options, masking, rng.as_mut())?;
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
    let (d, width, kv_width) = (grad_bo.data.len(), grad_bq.data.len(), grad_bk.data.len());
    let (examples, tokens) = examples_and_tokens(delta_batch, d, "delta_batch", context)?;
    let rows = examples * tokens;
    require_shape(x, delta_batch.shape, "x", context)?;
    for (cache, name) in [(h, "h"), (dq, "dq")] {
        require_shape(cache, Shape::Matrix(rows, width), name, context)?;
    }
    for (cache, name) in [(dk, "dk"), (dv, "dv")] {
        require_shape(cache, Shape::Matrix(rows, kv_width), name, context)?;
    }
    let x_rows = token_rows(x, d);
    let delta = token_rows(delta_batch, d);
    accumulate(
        [&delta, &x_rows, &x_rows, h],
        [dq, dk, dv],
        [grad_wq, grad_bq, grad_wk, grad_bk, grad_wv, grad_bv, grad_wo, grad_bo],
    )
}

/// The eight gradients after one batch's: `grad_wq`, `grad_bq` from `dq` against `x_rows`; `grad_wk`
/// ... `grad_bv` from `dk`, `dv` against `kv_rows`; `grad_wo`, `grad_bo` from `delta` against `h`.
#[allow(clippy::type_complexity)]
fn accumulate(
    rows: [&Array; 4],
    gradients: [&Array; 3],
    grads: [&Array; 8],
) -> Result<(Array, Array, Array, Array, Array, Array, Array, Array), Error> {
    let [delta, x_rows, kv_rows, h] = rows;
    let [dq, dk, dv] = gradients;
    let [grad_wq, grad_bq, grad_wk, grad_bk, grad_wv, grad_bv, grad_wo, grad_bo] = grads;
    let (grad_wq, grad_bq) = layer_accumulate_gradient_batch(dq, x_rows, grad_wq, grad_bq)?;
    let (grad_wk, grad_bk) = layer_accumulate_gradient_batch(dk, kv_rows, grad_wk, grad_bk)?;
    let (grad_wv, grad_bv) = layer_accumulate_gradient_batch(dv, kv_rows, grad_wv, grad_bv)?;
    let (grad_wo, grad_bo) = layer_accumulate_gradient_batch(delta, h, grad_wo, grad_bo)?;
    Ok((grad_wq, grad_bq, grad_wk, grad_bk, grad_wv, grad_bv, grad_wo, grad_bo))
}

// cross-attention (indrajala-ml's cross-attention workplan, D3, D11, D13): the queries from the
// decoder's tokens, the keys and values from the encoder's output, its memory

/// `d_m`, the memory's width, `wk`'s columns.
fn memory_width(wk: &Array, context: &str) -> Result<usize, Error> {
    match wk.shape {
        Shape::Matrix(_, d_m) if d_m > 0 => Ok(d_m),
        shape => Err(Error::invalid(format!(
            "{context} requires a 2D wk of the memory's width, got shape {shape:?}"
        ))),
    }
}

/// `x` and `memory` 2D, one row an example in each: `(examples, tokens, memory_tokens)`.
fn cross_extent(
    x: &Array,
    memory: &Array,
    (d, d_m): (usize, usize),
    context: &str,
) -> Result<(usize, usize, usize), Error> {
    for (array, name) in [(x, "x"), (memory, "memory")] {
        if !matches!(array.shape, Shape::Matrix(_, _)) {
            return Err(Error::invalid(format!(
                "{context} requires a 2D {name}, got shape {:?}",
                array.shape
            )));
        }
    }
    let (examples, tokens) = examples_and_tokens(x, d, "x", context)?;
    let (memory_examples, memory_tokens) = examples_and_tokens(memory, d_m, "memory", context)?;
    if memory_examples != examples {
        return Err(Error::invalid(format!(
            "{context} requires one memory row per example, {examples}; got {memory_examples}"
        )));
    }
    Ok((examples, tokens, memory_tokens))
}

/// The queries' and the keys' segment ids, both or neither, one per token of each side.
fn cross_segments<'a>(
    segments: Option<&'a Array>,
    memory_segments: Option<&'a Array>,
    (examples, tokens, memory_tokens): (usize, usize, usize),
    context: &str,
) -> Result<Option<(&'a Array, &'a Array)>, Error> {
    require_segments(segments, examples, tokens, context)?;
    require_segments(memory_segments, examples, memory_tokens, context)?;
    match (segments, memory_segments) {
        (None, None) => Ok(None),
        (Some(q), Some(k)) => Ok(Some((q, k))),
        _ => Err(Error::invalid(format!(
            "{context} requires both segments and memory_segments, or neither"
        ))),
    }
}

/// `CrossAttentionArrayLayer.forward_batch`: `x` 2D (`batch, T * d`), the decoder's tokens, `d`
/// `bo`'s length, attending to `memory` 2D (`batch, S * d_m`), the encoder's output, `d_m` `wk`'s
/// columns: `Q = X Wq^T + bq`, `K = M Wk^T + bk`, `V = M Wv^T + bv`, then attend and combine as
/// self-attention's, over `T` queries and `S` keys. Never causal, never rotary. With `segments`
/// (`N * T` ids) and `memory_segments` (`N * S`), both or neither, a query reads only the keys of
/// its own segment and never a pad's (0), with no own-column exception: a pad's query reads no key,
/// its `H` 0 and its output `bo`. Dropped from `rng` in `training` at `dropout > 0` as
/// self-attention is (per key block, every example's, head's, query's and key's draw, numpy's `(N,
/// h, T, b)`), tiled in key blocks of `block`. Returns `(a, q, k, v, h, m, l, dropout_rng)`: `q`,
/// `h`, `m`, `l` over the `(N * T)` query rows and `k`, `v` over the `(N * S)` memory rows.
#[allow(clippy::too_many_arguments)]
pub fn cross_attention_forward_batch(
    x: &Array,
    memory: &Array,
    wq: &Array,
    bq: &Array,
    wk: &Array,
    bk: &Array,
    wv: &Array,
    bv: &Array,
    wo: &Array,
    bo: &Array,
    heads: usize,
    key_value_heads: usize,
    dropout: f64,
    training: bool,
    rng: Option<&mut Generator>,
    segments: Option<&Array>,
    memory_segments: Option<&Array>,
    block: usize,
) -> Result<ForwardResult, Error> {
    let context = "cross_attention_forward_batch";
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
    let mut dropping = Dropping {
        dropout,
        training,
        rng,
        context,
    };
    let d_m = memory_width(wk, context)?;
    let (d, options) = shapes(&projections, (heads, key_value_heads), dropout, Some(d_m), context)?;
    require_block(block, context)?;
    let sides = cross_extent(x, memory, (d, d_m), context)?;
    let (examples, tokens, memory_tokens) = sides;
    let segments = cross_segments(segments, memory_segments, sides, context)?;
    let (q, k, v) = project(
        &token_rows(x, d),
        &token_rows(memory, d_m),
        &projections,
        None,
        tokens,
        &options,
    )?;
    // a copy in the state before the draws, for the backward op (D12); the tiles draw from the caller's
    let saved = dropping.generator()?.map(|rng| rng.clone());
    let extent = Extent {
        examples,
        queries: tokens,
        keys: memory_tokens,
        block,
        origin: 0,
    };
    let masking = cross_masking(segments, tokens, memory_tokens);
    let (h, m, l) = tiled_forward([&q, &k, &v], extent, &options, masking, dropping.generator()?)?;
    let out = combine(&h, &projections)?;
    let a = Array {
        data: out.data,
        shape: x.shape,
    };
    Ok((a, q, k, v, h, m, l, saved))
}

/// `CrossAttentionArrayLayer._backward`: from `delta_batch` (`dl/dout`, 2D `batch, T * d`) and the
/// forward pass's `q`, `k`, `v`, `h`, `m` and `l` (`cross_attention_forward_batch`'s), masked as the
/// forward was (`segments`, `memory_segments`) in its key blocks of `block`, its dropout drawn again
/// from a copy of `dropout_rng`, returns `(dx, dm, dq, dk, dv)`: `dx = dQ Wq` in `delta_batch`'s
/// shape; `dm = dK Wk + dV Wv`, the memory's gradient, `(N * S, d_m)` rows, as numpy's
/// `memory_downstream_batch`; and `dq` `(N * T, h·d_k)`, `dk`, `dv` `(N * S, g·d_k)`, which the
/// gradients read.
#[allow(clippy::too_many_arguments)]
pub fn cross_attention_downstream_batch(
    delta_batch: &Array,
    wq: &Array,
    wk: &Array,
    wv: &Array,
    wo: &Array,
    caches: [&Array; 6],
    heads: usize,
    key_value_heads: usize,
    segments: Option<&Array>,
    memory_segments: Option<&Array>,
    dropout: f64,
    dropout_rng: Option<&Generator>,
    block: usize,
) -> Result<(Array, Array, Array, Array, Array), Error> {
    let context = "cross_attention_downstream_batch";
    let [q, k, v, h, m, l] = caches;
    let width = match wq.shape {
        Shape::Matrix(width, _) => width,
        Shape::Vector(_) => 0,
    };
    let options = AttentionOptions::new(heads, key_value_heads, width, dropout, context)?;
    let d = columns(wq);
    let d_m = memory_width(wk, context)?;
    require_weights([wq, wk, wv, wo], (d, d_m), &options, context)?;
    require_block(block, context)?;
    if !matches!(delta_batch.shape, Shape::Matrix(_, _)) {
        return Err(Error::invalid(format!(
            "{context} requires a 2D delta_batch, got shape {:?}",
            delta_batch.shape
        )));
    }
    let (examples, tokens) = examples_and_tokens(delta_batch, d, "delta_batch", context)?;
    let rows = examples * tokens;
    let memory_rows = match k.shape {
        Shape::Matrix(memory_rows, _) if memory_rows > 0 && memory_rows.is_multiple_of(examples) => memory_rows,
        shape => {
            return Err(Error::invalid(format!(
                "{context} requires k of whole memories, S rows per example of {examples}; got shape {shape:?}"
            )));
        }
    };
    let memory_tokens = memory_rows / examples;
    for (cache, name) in [(q, "q"), (h, "h")] {
        require_shape(cache, Shape::Matrix(rows, width), name, context)?;
    }
    for (cache, name) in [(k, "k"), (v, "v")] {
        require_shape(cache, Shape::Matrix(memory_rows, options.kv_width()), name, context)?;
    }
    for (cache, name) in [(m, "m"), (l, "l")] {
        require_shape(cache, Shape::Matrix(rows, heads), name, context)?;
    }
    let segments = cross_segments(segments, memory_segments, (examples, tokens, memory_tokens), context)?;

    let dh = combine_backward(&token_rows(delta_batch, d), wo)?;
    let extent = Extent {
        examples,
        queries: tokens,
        keys: memory_tokens,
        block,
        origin: 0,
    };
    let masking = cross_masking(segments, tokens, memory_tokens);
    let mut rng = if dropout > 0.0 { dropout_rng.cloned() } else { None };
    let (dq, dk, dv) = tiled_backward(&dh, [q, k, v, h, m, l], extent, &options, masking, rng.as_mut())?;
    let dx = layer_downstream_batch(wq, &dq)?;
    let dx = Array {
        data: dx.data,
        shape: delta_batch.shape,
    };
    let dm_k = layer_downstream_batch(wk, &dk)?;
    let dm_v = layer_downstream_batch(wv, &dv)?;
    let dm = Array {
        data: dm_k.data.iter().zip(&dm_v.data).map(|(&a, &b)| a + b).collect(),
        shape: dm_k.shape,
    };
    Ok((dx, dm, dq, dk, dv))
}

/// `CrossAttentionArrayLayer.accumulate_gradient_batch`: each projection's `(grad_W, grad_b)` as a
/// dense layer's, `grad_W += delta^T X` and `grad_b += sum(delta)`: `dq` against `x`'s `(N * T, d)`
/// rows, `dk` and `dv` against `memory`'s `(N * S, d_m)` rows, and `delta_batch` against `h` (the
/// forward pass's), `delta_batch`, `x` and `memory` 2D. Returns the eight updated gradients in the
/// parameters' order.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn cross_attention_accumulate_gradient_batch(
    delta_batch: &Array,
    x: &Array,
    memory: &Array,
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
    let context = "cross_attention_accumulate_gradient_batch";
    let (d, width, kv_width) = (grad_bo.data.len(), grad_bq.data.len(), grad_bk.data.len());
    let d_m = memory_width(grad_wk, context)?;
    let (examples, tokens, memory_tokens) = cross_extent(x, memory, (d, d_m), context)?;
    require_shape(delta_batch, x.shape, "delta_batch", context)?;
    let (rows, memory_rows) = (examples * tokens, examples * memory_tokens);
    for (cache, name) in [(h, "h"), (dq, "dq")] {
        require_shape(cache, Shape::Matrix(rows, width), name, context)?;
    }
    for (cache, name) in [(dk, "dk"), (dv, "dv")] {
        require_shape(cache, Shape::Matrix(memory_rows, kv_width), name, context)?;
    }
    let (x_rows, memory_rows) = (token_rows(x, d), token_rows(memory, d_m));
    let delta = token_rows(delta_batch, d);
    accumulate(
        [&delta, &x_rows, &memory_rows, h],
        [dq, dk, dv],
        [grad_wq, grad_bq, grad_wk, grad_bk, grad_wv, grad_bv, grad_wo, grad_bo],
    )
}
