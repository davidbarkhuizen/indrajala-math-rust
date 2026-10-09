//! The token layers that aren't attention: a patch model's parameter-free ones (indrajala-ml's
//! layer-norm and attention workplan; its README, Layer norm and attention) and a sequence model's
//! `Embedding` (its sequence task workplan, D5). One Rust function per method of
//! `PatchesArrayLayer`, `TokenMeanArrayLayer` and `EmbeddingArrayLayer` in
//! `indrajala_ml/model/layers/numpy/token_array_layer.py`, as `fused.rs` is for `array_layer.py`.
//! A token sequence of `T` tokens of `d` features is flat and token-major, index `t * d + j`. Each
//! function takes one example (1D) or a batch (2D, one example per row) and returns the same rank.
//! `Position` needs none: it is `Array`'s `+` and `sum_axis0`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};

/// `(rows, width)` of one example (1 row) or a batch.
fn rows_and_width(x: &RustArray) -> (usize, usize) {
    match x.shape {
        Shape::Vector(n) => (1, n),
        Shape::Matrix(rows, n) => (rows, n),
    }
}

/// `data` (`rows` rows) back in `like`'s rank: 1D for one example, else `rows` by `width`.
fn same_rank(data: Vec<f64>, like: &RustArray, rows: usize, width: usize) -> RustArray {
    match like.shape {
        Shape::Vector(_) => RustArray::from_vector(data),
        Shape::Matrix(_, _) => RustArray::from_matrix(data, rows, width),
    }
}

/// The patch grid of an `(H, W, C)` image: `(H / p, W / p)` patches, checked to divide it.
struct PatchGrid {
    height: usize,
    width: usize,
    channels: usize,
    patch_size: usize,
}

impl PatchGrid {
    fn new(height: usize, width: usize, channels: usize, patch_size: usize, context: &str) -> PyResult<Self> {
        if patch_size == 0 || !height.is_multiple_of(patch_size) || !width.is_multiple_of(patch_size) {
            return Err(PyValueError::new_err(format!(
                "{context} requires a patch_size of 1 or more dividing the {height} x {width} image, got {patch_size}"
            )));
        }
        Ok(PatchGrid {
            height,
            width,
            channels,
            patch_size,
        })
    }

    fn size(&self) -> usize {
        self.height * self.width * self.channels
    }

    fn require_width(&self, x: &RustArray, name: &str, context: &str) -> PyResult<usize> {
        let (rows, n) = rows_and_width(x);
        if n != self.size() {
            return Err(PyValueError::new_err(format!(
                "{context} requires {name} of {} values per example ({} x {} x {}), got shape {:?}",
                self.size(),
                self.height,
                self.width,
                self.channels,
                x.shape
            )));
        }
        Ok(rows)
    }

    /// For each token-major index `t * k + f` in order, the channel-major image index it reads:
    /// token `t = u * (W / p) + v`, feature `f = c * p * p + i * p + j`, image `c * H * W + (u * p
    /// + i) * W + (v * p + j)`.
    fn image_indices(&self) -> Vec<usize> {
        let (h, w, p) = (self.height, self.width, self.patch_size);
        let mut indices = Vec::with_capacity(self.size());
        for u in 0..h / p {
            for v in 0..w / p {
                for c in 0..self.channels {
                    for i in 0..p {
                        for j in 0..p {
                            indices.push(c * h * w + (u * p + i) * w + (v * p + j));
                        }
                    }
                }
            }
        }
        indices
    }
}

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
    let context = "patches_forward";
    let grid = PatchGrid::new(height, width, channels, patch_size, context)?;
    let rows = grid.require_width(x, "x", context)?;
    let size = grid.size();
    let indices = grid.image_indices();
    let mut data = Vec::with_capacity(x.data.len());
    for image in x.data.chunks_exact(size.max(1)) {
        data.extend(indices.iter().map(|&index| image[index]));
    }
    Ok(same_rank(data, x, rows, size))
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
    let context = "patches_downstream";
    let grid = PatchGrid::new(height, width, channels, patch_size, context)?;
    let rows = grid.require_width(delta, "delta", context)?;
    let size = grid.size();
    let indices = grid.image_indices();
    let mut data = vec![0.0; delta.data.len()];
    for (image, tokens) in data
        .chunks_exact_mut(size.max(1))
        .zip(delta.data.chunks_exact(size.max(1)))
    {
        for (&index, &value) in indices.iter().zip(tokens) {
            image[index] = value;
        }
    }
    Ok(same_rank(data, delta, rows, size))
}

fn require_tokens(tokens: usize, width: usize, name: &str, context: &str) -> PyResult<usize> {
    if tokens == 0 || !width.is_multiple_of(tokens) {
        return Err(PyValueError::new_err(format!(
            "{context} requires {name} of tokens x features values per example, {tokens} tokens, got {width}"
        )));
    }
    Ok(width / tokens)
}

/// `TokenMeanArrayLayer.forward_batch` (and `forward`): `out_j = sum_t(x_tj) / T`, the sum a left
/// fold from `0.0` over the tokens in order, as numpy's `np.cumsum` along them.
#[pyfunction]
pub fn token_mean_forward(x: &RustArray, tokens: usize) -> PyResult<RustArray> {
    let (rows, width) = rows_and_width(x);
    let features = require_tokens(tokens, width, "x", "token_mean_forward")?;
    let mut data = Vec::with_capacity(rows * features);
    for example in x.data.chunks_exact(width.max(1)) {
        let mut sums = vec![0.0; features];
        for token in example.chunks_exact(features.max(1)) {
            for (acc, &v) in sums.iter_mut().zip(token) {
                *acc += v;
            }
        }
        data.extend(sums.iter().map(|&s| s / tokens as f64));
    }
    Ok(same_rank(data, x, rows, features))
}

/// `TokenMeanArrayLayer.downstream_batch` (and `downstream`): `dx_tj = delta_j / T` for every
/// token `t`.
#[pyfunction]
pub fn token_mean_downstream(delta: &RustArray, tokens: usize) -> PyResult<RustArray> {
    if tokens == 0 {
        return Err(PyValueError::new_err(
            "token_mean_downstream requires 1 or more tokens, got 0",
        ));
    }
    let (rows, features) = rows_and_width(delta);
    let mut data = Vec::with_capacity(rows * tokens * features);
    for example in delta.data.chunks_exact(features.max(1)) {
        let share: Vec<f64> = example.iter().map(|&v| v / tokens as f64).collect();
        for _ in 0..tokens {
            data.extend_from_slice(&share);
        }
    }
    Ok(same_rank(data, delta, rows, tokens * features))
}

/// `(vocabulary, size)` of an embedding table, checked to be a non-empty matrix.
fn table_shape(table: &RustArray, name: &str, context: &str) -> PyResult<(usize, usize)> {
    match table.shape {
        Shape::Matrix(vocabulary, size) if vocabulary > 0 && size > 0 => Ok((vocabulary, size)),
        shape => Err(PyValueError::new_err(format!(
            "{context} requires {name} of shape (vocabulary, size), both 1 or more, got {shape:?}"
        ))),
    }
}

/// `x`'s token ids, each refused unless a whole number in `[0, vocabulary)`, as
/// `EmbeddingArrayLayer._ids` refuses them.
fn token_ids(x: &RustArray, vocabulary: usize, context: &str) -> PyResult<Vec<usize>> {
    x.data
        .iter()
        .map(|&v| {
            if v >= 0.0 && v < vocabulary as f64 && v.fract() == 0.0 {
                Ok(v as usize)
            } else {
                Err(PyValueError::new_err(format!(
                    "{context} reads token ids, whole numbers in [0, {vocabulary}), got {v}"
                )))
            }
        })
        .collect()
}

/// `EmbeddingArrayLayer.forward_batch` (and `forward`): each of `x`'s `T` token ids per example as
/// its row of the `(vocabulary, size)` `table`, `T * size` values per example, copied.
#[pyfunction]
pub fn embedding_forward(x: &RustArray, table: &RustArray) -> PyResult<RustArray> {
    let context = "embedding_forward";
    let (vocabulary, size) = table_shape(table, "table", context)?;
    let (rows, tokens) = rows_and_width(x);
    let mut data = Vec::with_capacity(x.data.len() * size);
    for id in token_ids(x, vocabulary, context)? {
        data.extend_from_slice(&table.data[id * size..(id + 1) * size]);
    }
    Ok(same_rank(data, x, rows, tokens * size))
}

/// `EmbeddingArrayLayer.accumulate_gradient_batch`: `grad_table` with each of `delta`'s rows of
/// `size` added to the row its token id in `x` read, in row order (example by example, token by
/// token), as `np.add.at` adds them: per row of the table, a left fold onto its gradient. `delta`
/// is `T * size` values per example of `x`, in `x`'s rank. Returns the updated gradient.
#[pyfunction]
pub fn embedding_accumulate_gradient(delta: &RustArray, x: &RustArray, grad_table: &RustArray) -> PyResult<RustArray> {
    let context = "embedding_accumulate_gradient";
    let (vocabulary, size) = table_shape(grad_table, "grad_table", context)?;
    let (rows, tokens) = rows_and_width(x);
    let expected = match x.shape {
        Shape::Vector(_) => Shape::Vector(tokens * size),
        Shape::Matrix(_, _) => Shape::Matrix(rows, tokens * size),
    };
    if delta.shape != expected {
        return Err(PyValueError::new_err(format!(
            "{context} requires delta of {size} values per token id in x, x of shape {:?}, got {:?}",
            x.shape, delta.shape
        )));
    }
    let mut data = grad_table.data.clone();
    for (id, row) in token_ids(x, vocabulary, context)?
        .into_iter()
        .zip(delta.data.chunks_exact(size))
    {
        for (acc, &v) in data[id * size..(id + 1) * size].iter_mut().zip(row) {
            *acc += v;
        }
    }
    Ok(RustArray {
        data,
        shape: grad_table.shape,
    })
}
