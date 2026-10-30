//! The bindings of the core's `recurrent` (`indrajala_math::recurrent`): each `#[pyfunction]` lends its
//! arrays to the core's and moves the results in.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::linalg::py_err;
#[allow(unused_imports)]
pub(crate) use indrajala_math::recurrent::*;

/// `StateInputArrayLayer.forward_batch`: `out_t = x_t + P c` for each of the `tokens` tokens of
/// each example, `x` `(N, T * d)`, `state` `(N, s)`, `p` `(d, s)`; `P c` is
/// `linear_forward_batch(p, state)`.
#[pyfunction]
pub fn state_input_forward(x: &RustArray, state: &RustArray, p: &RustArray, tokens: usize) -> PyResult<RustArray> {
    indrajala_math::recurrent::state_input_forward(x, state, p, tokens)
        .map(RustArray::from)
        .map_err(py_err)
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
    indrajala_math::recurrent::state_input_backward(delta, state, p, grad_p, tokens)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

/// `AttentionPoolingArrayLayer.forward_batch`: the scores `s_t = q . x_t` (`x`'s `(N * T, d)` rows
/// `@ q`), `p` their `array_softmax` per example, and `u = p @ x` per example, `(1, T) @ (T, d)`.
/// Returns `(u, p)`, `(N, d)` and `(N, T)`.
#[pyfunction]
pub fn attention_pooling_forward(x: &RustArray, q: &RustArray, tokens: usize) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::recurrent::attention_pooling_forward(x, q, tokens)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
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
    indrajala_math::recurrent::attention_pooling_backward(x, p, q, du, grad_q, tokens)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
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
    indrajala_math::recurrent::gru_forward(u, c, w, u_weights, b)
        .map(|r| {
            let (r0, r1, r2, r3, r4) = r;
            (
                RustArray::from(r0),
                RustArray::from(r1),
                RustArray::from(r2),
                RustArray::from(r3),
                RustArray::from(r4),
            )
        })
        .map_err(py_err)
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
    indrajala_math::recurrent::gru_backward(dh, u, c, r, z, h_tilde, rc, w, u_weights, grad_w, grad_u, grad_b)
        .map(|r| {
            let (r0, r1, r2, r3, r4) = r;
            (
                RustArray::from(r0),
                RustArray::from(r1),
                RustArray::from(r2),
                RustArray::from(r3),
                RustArray::from(r4),
            )
        })
        .map_err(py_err)
}

/// `CodesArrayLayer.forward_batch`: each row of `h` snapped to its nearest row of `e` by squared
/// distance, a left fold over the features of `(h - e)^2`; the lowest index on a tie, and the
/// first NaN distance if any, as `np.argmin`. Returns `(q, indices)`, the codes, a copy, and
/// their indices.
#[pyfunction]
pub fn codes_forward(h: &RustArray, e: &RustArray) -> PyResult<(RustArray, Vec<usize>)> {
    indrajala_math::recurrent::codes_forward(h, e)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), r1)
        })
        .map_err(py_err)
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
    indrajala_math::recurrent::codes_backward(dq, h, q, indices, grad_e, weight, commitment, target.map(|a| &a.0))
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
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
    indrajala_math::recurrent::soft_codes_forward(h, e, temperature)
        .map(|r| {
            let (r0, r1, r2) = r;
            (RustArray::from(r0), RustArray::from(r1), r2)
        })
        .map_err(py_err)
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
    indrajala_math::recurrent::soft_codes_backward(
        dq,
        h,
        q,
        weights,
        e,
        grad_e,
        weight,
        commitment,
        temperature,
        straight,
    )
    .map(|r| {
        let (r0, r1) = r;
        (RustArray::from(r0), RustArray::from(r1))
    })
    .map_err(py_err)
}

/// `np.concatenate(arrays, axis=1)`: matrices of one row count side by side, a copy.
#[pyfunction]
pub fn concatenate_columns(arrays: Vec<PyRef<'_, RustArray>>) -> PyResult<RustArray> {
    indrajala_math::recurrent::concatenate_columns(&arrays.iter().map(|a| &a.0).collect::<Vec<_>>())
        .map(RustArray::from)
        .map_err(py_err)
}
