//! The bindings of the core's `nn::attention`: each `#[pyfunction]` lends its arrays to the core's
//! and moves the results in.

mod cross;

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::error::py_err;
use crate::random::generator::Generator;

/// `ForwardResult` (`indrajala_math::nn::attention`) with the bindings' arrays.
type ForwardResult = (
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    Option<Generator>,
);

/// The core's forward result, moved into the bindings' arrays.
fn forward_result(r: indrajala_math::nn::attention::ForwardResult) -> ForwardResult {
    let (a, q, k, v, h, m, l, rng) = r;
    (
        a.into(),
        q.into(),
        k.into(),
        v.into(),
        h.into(),
        m.into(),
        l.into(),
        rng.map(Generator),
    )
}

/// `AttentionArrayLayer.forward`: one example, `x` 1D of `T * d` values, `d` `bo`'s length, in
/// `heads` heads across `bq`'s length, masked if `causal`, its weights dropped from `rng` in
/// `training` at `dropout > 0`, its queries and keys rotated by `cos` and `sin` when given.
/// Tiled in key blocks of `block` (indrajala-ml's context-length workplan, D3). Returns `(a, q, k,
/// v, h, m, l, dropout_rng)`: `a` 1D, and what the backward pass reads, `q`, `k`, `v` and `h`
/// `(T, h·d_k)` (`k`, `v` `(T, g·d_k)`), `m` and `l` `(T, heads)`, and a copy of `rng` before the
/// draws, or `None`. `key_value_heads` `g` (default `heads`) divides `heads`: query head `i` reads
/// key/value head `i // (heads / g)` (indrajala-ml's grouped-query attention workplan, D4).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (x, wq, bq, wk, bk, wv, bv, wo, bo, heads, *, causal = false, dropout = 0.0, training = false, rng = None, cos = None, sin = None, segments = None, block = 16, key_value_heads = None))]
pub fn attention_forward<'py>(
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
    causal: bool,
    dropout: f64,
    training: bool,
    mut rng: Option<PyRefMut<'py, Generator>>,
    cos: Option<&RustArray>,
    sin: Option<&RustArray>,
    segments: Option<&RustArray>,
    block: usize,
    key_value_heads: Option<usize>,
) -> PyResult<ForwardResult> {
    indrajala_math::nn::attention::forward(
        x,
        wq,
        bq,
        wk,
        bk,
        wv,
        bv,
        wo,
        bo,
        heads,
        key_value_heads.unwrap_or(heads),
        causal,
        dropout,
        training,
        rng.as_deref_mut().map(|g| &mut g.0),
        cos.map(|a| &a.0),
        sin.map(|a| &a.0),
        segments.map(|a| &a.0),
        block,
    )
    .map(forward_result)
    .map_err(py_err)
}

/// `AttentionArrayLayer.forward_batch`: `x` 2D (`batch, T * d`), each example
/// masked if `causal` and by `segments`, dropped, rotated and tiled as `attention_forward`. Returns
/// `(a, q, k, v, h, m, l, dropout_rng)`, the caches over the batch's `(N * T)` rows, examples then
/// tokens; a key block's draws are every example's, head's, query's and key's in turn (D12).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (x, wq, bq, wk, bk, wv, bv, wo, bo, heads, *, causal = false, dropout = 0.0, training = false, rng = None, cos = None, sin = None, segments = None, block = 16, key_value_heads = None))]
pub fn attention_forward_batch<'py>(
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
    causal: bool,
    dropout: f64,
    training: bool,
    mut rng: Option<PyRefMut<'py, Generator>>,
    cos: Option<&RustArray>,
    sin: Option<&RustArray>,
    segments: Option<&RustArray>,
    block: usize,
    key_value_heads: Option<usize>,
) -> PyResult<ForwardResult> {
    indrajala_math::nn::attention::forward_batch(
        x,
        wq,
        bq,
        wk,
        bk,
        wv,
        bv,
        wo,
        bo,
        heads,
        key_value_heads.unwrap_or(heads),
        causal,
        dropout,
        training,
        rng.as_deref_mut().map(|g| &mut g.0),
        cos.map(|a| &a.0),
        sin.map(|a| &a.0),
        segments.map(|a| &a.0),
        block,
    )
    .map(forward_result)
    .map_err(py_err)
}

/// `AttentionArrayLayer.step`: one token's decode on a causal layer's cache, `x` 1D, `k_cache` and
/// `v_cache` `(c, g·d_k)` oldest first, `cos` and `sin` its position's `(1, d_k / 2)` row when
/// rotary, `origin` the first cached key's column after the append (the key blocks cut at the
/// columns that are multiples of `block`, as the full pass's). Returns `(a, k_cache, v_cache)`,
/// the caches with its key and value appended, the oldest dropped past `window`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (x, wq, bq, wk, bk, wv, bv, wo, bo, k_cache, v_cache, heads, *, window, cos = None, sin = None, origin = 0, block = 16, key_value_heads = None))]
pub fn attention_step(
    x: &RustArray,
    wq: &RustArray,
    bq: &RustArray,
    wk: &RustArray,
    bk: &RustArray,
    wv: &RustArray,
    bv: &RustArray,
    wo: &RustArray,
    bo: &RustArray,
    k_cache: &RustArray,
    v_cache: &RustArray,
    heads: usize,
    window: usize,
    cos: Option<&RustArray>,
    sin: Option<&RustArray>,
    origin: usize,
    block: usize,
    key_value_heads: Option<usize>,
) -> PyResult<(RustArray, RustArray, RustArray)> {
    indrajala_math::nn::attention::step(
        x,
        wq,
        bq,
        wk,
        bk,
        wv,
        bv,
        wo,
        bo,
        k_cache,
        v_cache,
        heads,
        key_value_heads.unwrap_or(heads),
        window,
        cos.map(|a| &a.0),
        sin.map(|a| &a.0),
        origin,
        block,
    )
    .map(|(a, k, v)| (a.into(), k.into(), v.into()))
    .map_err(py_err)
}

/// `AttentionArrayLayer._backward`: from `delta_batch` (`dl/dout`, 2D `batch, T * d`, or 1D for
/// one example, whose caches are `attention_forward`'s) and the forward pass's `q`, `k`, `v`, `h`,
/// `m` and `l`, in `heads` heads across `wq`'s rows, masked as the forward was (`causal`,
/// `segments`) in its key blocks of `block`, returns `(dx, dq, dk, dv)`: `dx`, the downstream, in
/// `delta_batch`'s shape, and `dq` `(N * T, h·d_k)` and `dk`, `dv` `(N * T, g·d_k)`, which the
/// gradients read.
/// A pass that dropped takes its `dropout` and the forward op's `dropout_rng`, from a copy of which
/// it draws the same masks again; one that rotated takes its `cos` and `sin`, and returns `dq` and
/// `dk` rotated back.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (delta_batch, wq, wk, wv, wo, q, k, v, h, m, l, heads, *, causal = false, segments = None, dropout = 0.0, dropout_rng = None, cos = None, sin = None, block = 16, key_value_heads = None))]
pub fn attention_downstream_batch(
    delta_batch: &RustArray,
    wq: &RustArray,
    wk: &RustArray,
    wv: &RustArray,
    wo: &RustArray,
    q: &RustArray,
    k: &RustArray,
    v: &RustArray,
    h: &RustArray,
    m: &RustArray,
    l: &RustArray,
    heads: usize,
    causal: bool,
    segments: Option<&RustArray>,
    dropout: f64,
    dropout_rng: Option<PyRef<'_, Generator>>,
    cos: Option<&RustArray>,
    sin: Option<&RustArray>,
    block: usize,
    key_value_heads: Option<usize>,
) -> PyResult<(RustArray, RustArray, RustArray, RustArray)> {
    indrajala_math::nn::attention::downstream_batch(
        delta_batch,
        wq,
        wk,
        wv,
        wo,
        [q, k, v, h, m, l],
        heads,
        key_value_heads.unwrap_or(heads),
        causal,
        segments.map(|a| &a.0),
        dropout,
        dropout_rng.as_deref().map(|g| &g.0),
        cos.map(|a| &a.0),
        sin.map(|a| &a.0),
        block,
    )
    .map(|(dx, dq, dk, dv)| (dx.into(), dq.into(), dk.into(), dv.into()))
    .map_err(py_err)
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
    indrajala_math::nn::attention::accumulate_gradient_batch(
        delta_batch,
        x,
        h,
        dq,
        dk,
        dv,
        grad_wq,
        grad_bq,
        grad_wk,
        grad_bk,
        grad_wv,
        grad_bv,
        grad_wo,
        grad_bo,
    )
    .map(|r| {
        let (r0, r1, r2, r3, r4, r5, r6, r7) = r;
        (
            RustArray::from(r0),
            RustArray::from(r1),
            RustArray::from(r2),
            RustArray::from(r3),
            RustArray::from(r4),
            RustArray::from(r5),
            RustArray::from(r6),
            RustArray::from(r7),
        )
    })
    .map_err(py_err)
}

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(attention_forward, m)?)?;
    m.add_function(wrap_pyfunction!(attention_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(attention_step, m)?)?;
    m.add_function(wrap_pyfunction!(attention_downstream_batch, m)?)?;
    m.add_function(wrap_pyfunction!(attention_accumulate_gradient_batch, m)?)?;
    cross::register(m)?;
    Ok(())
}
