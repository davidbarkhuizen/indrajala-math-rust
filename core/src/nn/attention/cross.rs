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

use super::*;

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

/// `CrossAttentionArrayLayer`'s decoding (indrajala-ml's cross-attention workplan, D7): the queries
/// `x` (2D, `batch, T * d`, a prefill's `T` tokens or a step's one) against keys `k` and values `v`
/// computed once per source, `(N * S, g·d_k)` each, the memory's rows through `wk`, `bk` and `wv`,
/// `bv` (`affine_forward_batch`, project's products): `Q = X Wq^T + bq`, then attend and combine as
/// `cross_attention_forward_batch`, masked by `segments` and `memory_segments` (both or neither),
/// never dropped. Returns `a` in `x`'s shape. Each query's row is attended alone in the same key
/// blocks, so its bits are the full pass's at that token.
#[allow(clippy::too_many_arguments)]
pub fn cross_attention_attend_batch(
    x: &Array,
    k: &Array,
    v: &Array,
    wq: &Array,
    bq: &Array,
    wo: &Array,
    bo: &Array,
    heads: usize,
    key_value_heads: usize,
    segments: Option<&Array>,
    memory_segments: Option<&Array>,
    block: usize,
) -> Result<Array, Error> {
    let context = "cross_attention_attend_batch";
    let (d, width) = (bo.data.len(), bq.data.len());
    if d == 0 {
        return Err(Error::invalid(format!("{context} requires a non-empty bo")));
    }
    let options = AttentionOptions::new(heads, key_value_heads, width, 0.0, context)?;
    require_shape(wq, Shape::Matrix(width, d), "wq", context)?;
    require_shape(bq, Shape::Vector(width), "bq", context)?;
    require_shape(wo, Shape::Matrix(d, width), "wo", context)?;
    require_block(block, context)?;
    if !matches!(x.shape, Shape::Matrix(_, _)) {
        return Err(Error::invalid(format!(
            "{context} requires a 2D x, got shape {:?}",
            x.shape
        )));
    }
    let (examples, tokens) = examples_and_tokens(x, d, "x", context)?;
    let memory_rows = match k.shape {
        Shape::Matrix(rows, _) if rows > 0 && rows.is_multiple_of(examples) => rows,
        shape => {
            return Err(Error::invalid(format!(
                "{context} requires k of whole memories, S rows per example of {examples}; got shape {shape:?}"
            )));
        }
    };
    require_shape(k, Shape::Matrix(memory_rows, options.kv_width()), "k", context)?;
    require_shape(v, Shape::Matrix(memory_rows, options.kv_width()), "v", context)?;
    let memory_tokens = memory_rows / examples;
    let segments = cross_segments(segments, memory_segments, (examples, tokens, memory_tokens), context)?;
    let q = affine_forward_batch(wq, &token_rows(x, d), bq)?;
    let extent = Extent {
        examples,
        queries: tokens,
        keys: memory_tokens,
        block,
        origin: 0,
    };
    let masking = cross_masking(segments, tokens, memory_tokens);
    let (h, _m, _l) = tiled_forward([&q, k, v], extent, &options, masking, None)?;
    let out = affine_forward_batch(wo, &h, bo)?;
    Ok(Array {
        data: out.data,
        shape: x.shape,
    })
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
