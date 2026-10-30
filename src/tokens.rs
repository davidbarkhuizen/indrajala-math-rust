//! The bindings of the core's `tokens` (`indrajala_math::tokens`): each `#[pyfunction]` lends its
//! arrays to the core's and moves the results in.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::generator::Generator;
use crate::linalg::py_err;
#[allow(unused_imports)]
pub(crate) use indrajala_math::tokens::*;

/// `PatchesArrayLayer.forward_batch` (and `forward`, one example): each channel-major `(H, W, C)`
/// image as `H/p * W/p` tokens of `p * p * C` features, a fixed permutation.
#[pyfunction]
pub fn patches_forward(
    x: &RustArray,
    height: usize,
    width: usize,
    channels: usize,
    patch_size: usize,
) -> PyResult<RustArray> {
    indrajala_math::tokens::patches_forward(x, height, width, channels, patch_size)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `PatchesArrayLayer.downstream_batch` (and `downstream`): the inverse permutation of `delta`,
/// each token-major value back at its image index.
#[pyfunction]
pub fn patches_downstream(
    delta: &RustArray,
    height: usize,
    width: usize,
    channels: usize,
    patch_size: usize,
) -> PyResult<RustArray> {
    indrajala_math::tokens::patches_downstream(delta, height, width, channels, patch_size)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `TokenMeanArrayLayer.forward_batch` (and `forward`): `out_j = sum_t(x_tj) / T`, the sum a left
/// fold from `0.0` over the tokens in order, as numpy's `np.cumsum` along them.
#[pyfunction]
pub fn token_mean_forward(x: &RustArray, tokens: usize) -> PyResult<RustArray> {
    indrajala_math::tokens::token_mean_forward(x, tokens)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `TokenMeanArrayLayer.downstream_batch` (and `downstream`): `dx_tj = delta_j / T` for every
/// token `t`.
#[pyfunction]
pub fn token_mean_downstream(delta: &RustArray, tokens: usize) -> PyResult<RustArray> {
    indrajala_math::tokens::token_mean_downstream(delta, tokens)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `EmbeddingArrayLayer.forward_batch` (and `forward`): each of `x`'s `T` token ids per example as
/// its row of the `(vocabulary, size)` `table`, `T * size` values per example, copied.
#[pyfunction]
pub fn embedding_forward(x: &RustArray, table: &RustArray) -> PyResult<RustArray> {
    indrajala_math::tokens::embedding_forward(x, table)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `EmbeddingArrayLayer.accumulate_gradient_batch`: `grad_table` with each of `delta`'s rows of
/// `size` added to the row its token id in `x` read, in row order (example by example, token by
/// token), as `np.add.at` adds them: per row of the table, a left fold onto its gradient. `delta`
/// is `T * size` values per example of `x`, in `x`'s rank. Returns the updated gradient.
#[pyfunction]
pub fn embedding_accumulate_gradient(delta: &RustArray, x: &RustArray, grad_table: &RustArray) -> PyResult<RustArray> {
    indrajala_math::tokens::embedding_accumulate_gradient(delta, x, grad_table)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `TokenDropoutArrayLayer.forward_batch` in training: a mask of `x`'s shape drawn from `rng` in
/// C order (example, token, feature), `u >= drop_probability`, then `x * mask / keep`. One
/// example (1D) or a batch (2D); an example's draws are its batch of one's. Returns `(a, mask)`.
/// In inference the layer passes its input on and calls nothing.
#[pyfunction]
pub fn token_dropout_forward(
    x: &RustArray,
    drop_probability: f64,
    mut rng: PyRefMut<'_, Generator>,
) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::tokens::token_dropout_forward(x, drop_probability, &mut rng.0)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

/// `TokenDropoutArrayLayer`'s backward pass after a training forward: `delta * mask / keep`, the
/// downstream, `mask` `token_dropout_forward`'s.
#[pyfunction]
pub fn token_dropout_downstream(delta: &RustArray, mask: &RustArray, drop_probability: f64) -> PyResult<RustArray> {
    indrajala_math::tokens::token_dropout_downstream(delta, mask, drop_probability)
        .map(RustArray::from)
        .map_err(py_err)
}
