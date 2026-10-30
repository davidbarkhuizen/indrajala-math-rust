//! The bindings of the core's `attention` (`indrajala_math::attention`): each `#[pyfunction]` lends its
//! arrays to the core's and moves the results in.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::generator::Generator;
use crate::linalg::py_err;

/// `ForwardResult` (`indrajala_math::attention`) with the bindings' arrays.
type ForwardResult = (
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    RustArray,
    Option<RustArray>,
);

/// The core's forward result, moved into the bindings' arrays.
fn forward_result(r: indrajala_math::attention::ForwardResult) -> ForwardResult {
    let (a, b, c, d, e, f, g) = r;
    (
        a.into(),
        b.into(),
        c.into(),
        d.into(),
        e.into(),
        f.into(),
        g.map(RustArray::from),
    )
}
#[allow(unused_imports)]
pub(crate) use indrajala_math::attention::*;

/// `AttentionArrayLayer.forward`: one example, `x` 1D of `T * d` values, `d` `bo`'s length, in
/// `heads` heads across `bq`'s length, masked if `causal`, its weights dropped from `rng` in
/// `training` at `dropout > 0`, its queries and keys rotated by `cos` and `sin` when given.
/// Returns `(a, q, k, v, p, h, mask)`: `a` 1D, and what the backward
/// pass reads, `q`, `k`, `v` and `h` `(T, h·d_k)`, `p` `(T, h·T)` and `mask` as `p`, or `None`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (x, wq, bq, wk, bk, wv, bv, wo, bo, heads, *, causal = false, dropout = 0.0, training = false, rng = None, cos = None, sin = None))]
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
) -> PyResult<ForwardResult> {
    indrajala_math::attention::attention_forward(
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
        causal,
        dropout,
        training,
        rng.as_deref_mut().map(|g| &mut g.0),
        cos.map(|a| &a.0),
        sin.map(|a| &a.0),
    )
    .map(forward_result)
    .map_err(py_err)
}

/// `AttentionArrayLayer.forward_batch`: `x` 2D (`batch, T * d`), each example
/// `attention_forward`'s bits, masked if `causal`, dropped and rotated as it. Returns `(a, q, k, v, p, h,
/// mask)`, the caches over the batch's `(N * T)` rows, examples then tokens; the mask's draws are
/// the examples' in turn, so one example's are its batch of one's.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (x, wq, bq, wk, bk, wv, bv, wo, bo, heads, *, causal = false, dropout = 0.0, training = false, rng = None, cos = None, sin = None))]
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
) -> PyResult<ForwardResult> {
    indrajala_math::attention::attention_forward_batch(
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
        causal,
        dropout,
        training,
        rng.as_deref_mut().map(|g| &mut g.0),
        cos.map(|a| &a.0),
        sin.map(|a| &a.0),
    )
    .map(forward_result)
    .map_err(py_err)
}

/// `AttentionArrayLayer._backward`: from `delta_batch` (`dl/dout`, 2D `batch, T * d`, or 1D for
/// one example, whose caches are `attention_forward`'s) and the forward pass's `q`, `k`, `v` and
/// `p`, in `heads` heads across `wq`'s rows, returns `(dx, dq, dk, dv)`: `dx`, the downstream, in
/// `delta_batch`'s shape, and `dq`, `dk`, `dv` as `(N * T, h·d_k)` rows, which the gradients read.
/// A pass that dropped takes its `mask` (the forward op's, shaped as `p`) and its `dropout`; one
/// that rotated takes its `cos` and `sin`, and returns `dq` and `dk` rotated back.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (delta_batch, wq, wk, wv, wo, q, k, v, p, heads, *, mask = None, dropout = 0.0, cos = None, sin = None))]
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
    mask: Option<&RustArray>,
    dropout: f64,
    cos: Option<&RustArray>,
    sin: Option<&RustArray>,
) -> PyResult<(RustArray, RustArray, RustArray, RustArray)> {
    indrajala_math::attention::attention_downstream_batch(
        delta_batch,
        wq,
        wk,
        wv,
        wo,
        q,
        k,
        v,
        p,
        heads,
        mask.map(|a| &a.0),
        dropout,
        cos.map(|a| &a.0),
        sin.map(|a| &a.0),
    )
    .map(|r| {
        let (r0, r1, r2, r3) = r;
        (
            RustArray::from(r0),
            RustArray::from(r1),
            RustArray::from(r2),
            RustArray::from(r3),
        )
    })
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
    indrajala_math::attention::attention_accumulate_gradient_batch(
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
