//! The bindings of the core's `nn::attention::cross`: cross-attention's passes.

use super::*;

/// `CrossAttentionArrayLayer.forward_batch` (indrajala-ml's cross-attention workplan, D3): `x` 2D
/// (`batch, T * d`), the decoder's tokens, attending to `memory` 2D (`batch, S * d_m`), the
/// encoder's output, `d_m` `wk`'s columns. Never causal, never rotary; with `segments` and
/// `memory_segments`, both or neither, a query reads only its own segment's keys and never a pad's,
/// with no own-column exception, so a pad's query reads none (its `h` 0). Dropped and tiled as
/// `attention_forward_batch`. Returns `(a, q, k, v, h, m, l, dropout_rng)`, `k` and `v` over the
/// `(N * S)` memory rows.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (x, memory, wq, bq, wk, bk, wv, bv, wo, bo, heads, *, dropout = 0.0, training = false, rng = None, segments = None, memory_segments = None, block = 16, key_value_heads = None))]
pub fn cross_attention_forward_batch<'py>(
    x: &RustArray,
    memory: &RustArray,
    wq: &RustArray,
    bq: &RustArray,
    wk: &RustArray,
    bk: &RustArray,
    wv: &RustArray,
    bv: &RustArray,
    wo: &RustArray,
    bo: &RustArray,
    heads: usize,
    dropout: f64,
    training: bool,
    mut rng: Option<PyRefMut<'py, Generator>>,
    segments: Option<&RustArray>,
    memory_segments: Option<&RustArray>,
    block: usize,
    key_value_heads: Option<usize>,
) -> PyResult<ForwardResult> {
    indrajala_math::nn::attention::cross::forward_batch(
        x,
        memory,
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
        dropout,
        training,
        rng.as_deref_mut().map(|g| &mut g.0),
        segments.map(|a| &a.0),
        memory_segments.map(|a| &a.0),
        block,
    )
    .map(forward_result)
    .map_err(py_err)
}

/// `CrossAttentionArrayLayer._backward`: from `delta_batch` (2D) and the forward pass's caches,
/// masked and dropped as the forward was, returns `(dx, dm, dq, dk, dv)`: `dx = dQ Wq` in
/// `delta_batch`'s shape, `dm = dK Wk + dV Wv` the memory's gradient as `(N * S, d_m)` rows, and
/// `dq`, `dk`, `dv`, which the gradients read.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (delta_batch, wq, wk, wv, wo, q, k, v, h, m, l, heads, *, segments = None, memory_segments = None, dropout = 0.0, dropout_rng = None, block = 16, key_value_heads = None))]
pub fn cross_attention_downstream_batch(
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
    segments: Option<&RustArray>,
    memory_segments: Option<&RustArray>,
    dropout: f64,
    dropout_rng: Option<PyRef<'_, Generator>>,
    block: usize,
    key_value_heads: Option<usize>,
) -> PyResult<(RustArray, RustArray, RustArray, RustArray, RustArray)> {
    indrajala_math::nn::attention::cross::downstream_batch(
        delta_batch,
        wq,
        wk,
        wv,
        wo,
        [q, k, v, h, m, l],
        heads,
        key_value_heads.unwrap_or(heads),
        segments.map(|a| &a.0),
        memory_segments.map(|a| &a.0),
        dropout,
        dropout_rng.as_deref().map(|g| &g.0),
        block,
    )
    .map(|(dx, dm, dq, dk, dv)| (dx.into(), dm.into(), dq.into(), dk.into(), dv.into()))
    .map_err(py_err)
}

/// `CrossAttentionArrayLayer.accumulate_gradient_batch`: each projection's `(grad_W, grad_b)` as a
/// dense layer's, `dq` against `x`'s rows, `dk` and `dv` against `memory`'s, `delta_batch` against
/// `h`. Returns the eight updated gradients in the parameters' order.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
pub fn cross_attention_accumulate_gradient_batch(
    delta_batch: &RustArray,
    x: &RustArray,
    memory: &RustArray,
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
    indrajala_math::nn::attention::cross::accumulate_gradient_batch(
        delta_batch,
        x,
        memory,
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
    .map(|(r0, r1, r2, r3, r4, r5, r6, r7)| {
        (
            r0.into(),
            r1.into(),
            r2.into(),
            r3.into(),
            r4.into(),
            r5.into(),
            r6.into(),
            r7.into(),
        )
    })
    .map_err(py_err)
}

/// `CrossAttentionArrayLayer`'s decoding (indrajala-ml's cross-attention workplan, D7): the queries
/// `x` (2D) against keys `k` and values `v` computed once per source, `(N * S, g·d_k)` each, masked
/// by `segments` and `memory_segments` (both or neither), never dropped. Returns `a` in `x`'s shape,
/// each query's bits the full pass's at that token.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (x, k, v, wq, bq, wo, bo, heads, *, segments = None, memory_segments = None, block = 16, key_value_heads = None))]
pub fn cross_attention_attend_batch(
    x: &RustArray,
    k: &RustArray,
    v: &RustArray,
    wq: &RustArray,
    bq: &RustArray,
    wo: &RustArray,
    bo: &RustArray,
    heads: usize,
    segments: Option<&RustArray>,
    memory_segments: Option<&RustArray>,
    block: usize,
    key_value_heads: Option<usize>,
) -> PyResult<RustArray> {
    indrajala_math::nn::attention::cross::attend_batch(
        x,
        k,
        v,
        wq,
        bq,
        wo,
        bo,
        heads,
        key_value_heads.unwrap_or(heads),
        segments.map(|a| &a.0),
        memory_segments.map(|a| &a.0),
        block,
    )
    .map(RustArray::from)
    .map_err(py_err)
}

/// Adds this module's functions and classes to the Python module, then its children's.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cross_attention_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cross_attention_downstream_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cross_attention_accumulate_gradient_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cross_attention_attend_batch, m)?)?;
    Ok(())
}
