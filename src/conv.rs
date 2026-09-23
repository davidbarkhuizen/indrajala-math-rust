use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};
use crate::linalg::matmul;

/// Convolution and max pooling ops, one Rust call per `ConvArrayLayer`/`MaxPoolArrayLayer` method
/// (indrajala-ml's `indrajala_ml/model/conv_array_layer.py`/`max_pool_array_layer.py`),
/// batch-only: the Python layer wraps a single example as a batch of one. `RustArray` stays
/// 1D/2D, so every tensor crosses the boundary as a matrix and the 4D views exist only as index
/// arithmetic here:
///
/// - activations and deltas are `(N, C*H*W)`, channel-major (flat index `c*H*W + r*W + col`);
/// - the kernel matrix `W` is `(channel_count, C*k*k)`, each row in (channel, kernel row, kernel
///   col) order;
/// - the cached im2col columns are `(N*P, C*k*k)`, `P = out_height * out_width`, row `n*P + p`
///   with `p` in row-major output order, columns in `W`-row order.
///
/// Every conv reduction goes through `linalg::matmul`, so the summation order is that file's one
/// fixed grouping - machine-independent, and within `rtol` of numpy rather than bit-identical to
/// it. Pooling does no arithmetic beyond the scatter-add, so it matches numpy exactly.

/// The shape arithmetic for one conv or pool layer, built once by the Python layer and passed to
/// every call. Pooling uses it with `kernel_size = pool_size`.
#[pyclass(frozen)]
#[derive(Clone, Copy, Debug)]
pub struct ConvGeometry {
    #[pyo3(get)]
    pub input_height: usize,
    #[pyo3(get)]
    pub input_width: usize,
    #[pyo3(get)]
    pub input_channels: usize,
    #[pyo3(get)]
    pub kernel_size: usize,
    #[pyo3(get)]
    pub stride: usize,
    #[pyo3(get)]
    pub out_height: usize,
    #[pyo3(get)]
    pub out_width: usize,
    #[pyo3(get)]
    pub positions: usize,
    #[pyo3(get)]
    pub fan_in: usize,
    #[pyo3(get)]
    pub input_size: usize,
}

#[pymethods]
impl ConvGeometry {
    /// The same checks as `ConvArrayLayer`'s constructor, raised as `ValueError`.
    #[new]
    fn new(
        input_height: usize,
        input_width: usize,
        input_channels: usize,
        kernel_size: usize,
        stride: usize,
    ) -> PyResult<Self> {
        if input_channels < 1 {
            return Err(PyValueError::new_err("input_channels must be at least 1; got 0"));
        }
        if kernel_size < 1 {
            return Err(PyValueError::new_err("kernel_size must be at least 1; got 0"));
        }
        if stride < 1 {
            return Err(PyValueError::new_err("stride must be at least 1; got 0"));
        }
        if kernel_size > input_height || kernel_size > input_width {
            return Err(PyValueError::new_err(format!(
                "kernel_size ({kernel_size}) must fit within input_height x input_width \
                 ({input_height}x{input_width})"
            )));
        }
        let out_height = (input_height - kernel_size) / stride + 1;
        let out_width = (input_width - kernel_size) / stride + 1;
        Ok(ConvGeometry {
            input_height,
            input_width,
            input_channels,
            kernel_size,
            stride,
            out_height,
            out_width,
            positions: out_height * out_width,
            fan_in: input_channels * kernel_size * kernel_size,
            input_size: input_channels * input_height * input_width,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ConvGeometry(input_height={}, input_width={}, input_channels={}, kernel_size={}, stride={})",
            self.input_height, self.input_width, self.input_channels, self.kernel_size, self.stride
        )
    }
}

impl ConvGeometry {
    /// The flat index into one example's `C*H*W` input that kernel offset `(kr, kc)` of channel
    /// `c` reads at output position `(out_row, out_col)`.
    #[inline]
    pub(crate) fn input_index(&self, c: usize, out_row: usize, out_col: usize, kr: usize, kc: usize) -> usize {
        let row = out_row * self.stride + kr;
        let col = out_col * self.stride + kc;
        (c * self.input_height + row) * self.input_width + col
    }
}

pub(crate) fn require_matrix(arr: &RustArray, rows: Option<usize>, cols: usize, context: &str) -> PyResult<usize> {
    match arr.shape {
        Shape::Matrix(r, c) if c == cols && rows.map_or(true, |expected| expected == r) => Ok(r),
        shape => Err(PyValueError::new_err(format!(
            "{context}: expected a matrix of shape ({}, {cols}), got {:?}",
            rows.map_or("N".to_string(), |r| r.to_string()),
            shape
        ))),
    }
}

/// `W`'s row count (the output channel count), after checking its columns match the geometry's
/// kernel fan-in.
fn channel_count(w: &RustArray, geometry: &ConvGeometry, context: &str) -> PyResult<usize> {
    match w.shape {
        Shape::Matrix(rows, cols) if cols == geometry.fan_in => Ok(rows),
        shape => Err(PyValueError::new_err(format!(
            "{context}: W must be (channel_count, {}), got {:?}",
            geometry.fan_in, shape
        ))),
    }
}

/// `(N, O*P)` channel-major deltas to `(N*P, O)` - one row per output position, the layout
/// `cols @ W.T` produced them in.
fn deltas_by_position(delta: &RustArray, n: usize, o: usize, p: usize) -> RustArray {
    let mut out = vec![0.0; n * p * o];
    for example in 0..n {
        for channel in 0..o {
            let src = &delta.data[(example * o + channel) * p..(example * o + channel + 1) * p];
            for (position, &value) in src.iter().enumerate() {
                out[(example * p + position) * o + channel] = value;
            }
        }
    }
    RustArray::from_matrix(out, n * p, o)
}

/// `(N, O*P)` channel-major deltas to `(O, N*P)` - each channel's deltas over every example and
/// position in one row, the left operand of the kernel-gradient matmul.
fn deltas_by_channel(delta: &RustArray, n: usize, o: usize, p: usize) -> RustArray {
    let mut out = vec![0.0; o * n * p];
    for example in 0..n {
        for channel in 0..o {
            let src = &delta.data[(example * o + channel) * p..(example * o + channel + 1) * p];
            let dst = channel * n * p + example * p;
            out[dst..dst + p].copy_from_slice(src);
        }
    }
    RustArray::from_matrix(out, o, n * p)
}

/// `ConvArrayLayer.forward_batch`: im2col, `cols @ W.T` with `matmul`, then a scatter into
/// channel-major `(N, O*P)` that adds `b` and applies the ReLU in the same pass. Returns `(A,
/// cols)`: `cols` is kept by the caller for `conv_accumulate_gradient_batch`, as
/// `ConvArrayLayer._cols` is. The pre-activation `Z` is never stored: nothing in the backward
/// pass reads it (`array_relu_mask` masks on `A`).
#[pyfunction]
pub fn conv_forward_batch(
    w: &RustArray,
    x: &RustArray,
    b: &RustArray,
    geometry: &ConvGeometry,
) -> PyResult<(RustArray, RustArray)> {
    let o = channel_count(w, geometry, "conv_forward_batch")?;
    if b.shape != Shape::Vector(o) {
        return Err(PyValueError::new_err(format!(
            "conv_forward_batch: b must be a vector of length {o}, got {:?}",
            b.shape
        )));
    }
    let n = require_matrix(x, None, geometry.input_size, "conv_forward_batch X")?;
    let g = geometry;
    let (p, k, fan_in) = (g.positions, g.kernel_size, g.fan_in);

    let mut cols = vec![0.0; n * p * fan_in];
    for example in 0..n {
        let input = &x.data[example * g.input_size..(example + 1) * g.input_size];
        for out_row in 0..g.out_height {
            for out_col in 0..g.out_width {
                let row = &mut cols[(example * p + out_row * g.out_width + out_col) * fan_in..][..fan_in];
                let mut column = 0;
                for c in 0..g.input_channels {
                    for kr in 0..k {
                        let start = g.input_index(c, out_row, out_col, kr, 0);
                        row[column..column + k].copy_from_slice(&input[start..start + k]);
                        column += k;
                    }
                }
            }
        }
    }
    let cols = RustArray::from_matrix(cols, n * p, fan_in);

    let by_position = matmul(&cols, &w.transpose())?; // (N*P, O)
    let mut a = vec![0.0; n * o * p];
    for example in 0..n {
        for position in 0..p {
            let src = &by_position.data[(example * p + position) * o..][..o];
            for channel in 0..o {
                a[(example * o + channel) * p + position] = (src[channel] + b.data[channel]).max(0.0);
            }
        }
    }
    Ok((RustArray::from_matrix(a, n, o * p), cols))
}

/// `ConvArrayLayer._downstream`: `dcols = D @ W` with `matmul`, `D` the deltas regrouped to
/// `(N*P, O)`, then col2im as a scatter-add. The loop runs kernel offset `(kr, kc)` outside
/// output position, so each input accumulates its contributions in the same order as numpy's
/// one-slice-add-per-offset loop.
#[pyfunction]
pub fn conv_downstream_batch(w: &RustArray, delta_batch: &RustArray, geometry: &ConvGeometry) -> PyResult<RustArray> {
    let o = channel_count(w, geometry, "conv_downstream_batch")?;
    let g = geometry;
    let (p, k, fan_in) = (g.positions, g.kernel_size, g.fan_in);
    let n = require_matrix(delta_batch, None, o * p, "conv_downstream_batch delta_batch")?;

    let dcols = matmul(&deltas_by_position(delta_batch, n, o, p), w)?; // (N*P, C*k*k)
    let mut dx = vec![0.0; n * g.input_size];
    for example in 0..n {
        let out = &mut dx[example * g.input_size..(example + 1) * g.input_size];
        for c in 0..g.input_channels {
            for kr in 0..k {
                for kc in 0..k {
                    let column = (c * k + kr) * k + kc;
                    for out_row in 0..g.out_height {
                        for out_col in 0..g.out_width {
                            let position = out_row * g.out_width + out_col;
                            out[g.input_index(c, out_row, out_col, kr, kc)] +=
                                dcols.data[(example * p + position) * fan_in + column];
                        }
                    }
                }
            }
        }
    }
    Ok(RustArray::from_matrix(dx, n, g.input_size))
}

/// `ConvArrayLayer._accumulate`: `grad_W += D @ cols` with `matmul`, `D` the deltas regrouped to
/// `(O, N*P)`, and `grad_b += ` each channel's delta sum over every example and position. Returns
/// the updated pair for the caller to rebind, as `layer_accumulate_gradient_batch` does.
#[pyfunction]
pub fn conv_accumulate_gradient_batch(
    delta_batch: &RustArray,
    cols: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    geometry: &ConvGeometry,
) -> PyResult<(RustArray, RustArray)> {
    let o = channel_count(grad_w, geometry, "conv_accumulate_gradient_batch")?;
    if grad_b.shape != Shape::Vector(o) {
        return Err(PyValueError::new_err(format!(
            "conv_accumulate_gradient_batch: grad_b must be a vector of length {o}, got {:?}",
            grad_b.shape
        )));
    }
    let p = geometry.positions;
    let n = require_matrix(delta_batch, None, o * p, "conv_accumulate_gradient_batch delta_batch")?;
    require_matrix(cols, Some(n * p), geometry.fan_in, "conv_accumulate_gradient_batch cols")?;

    let by_channel = deltas_by_channel(delta_batch, n, o, p); // (O, N*P)
    let update = matmul(&by_channel, cols)?;
    let new_grad_w = grad_w.combine_with_array(&update, |g, u| g + u, "add")?;
    let new_grad_b = grad_b
        .data
        .iter()
        .enumerate()
        .map(|(channel, &gb)| gb + by_channel.data[channel * n * p..(channel + 1) * n * p].iter().sum::<f64>())
        .collect();
    Ok((new_grad_w, RustArray::from_vector(new_grad_b)))
}

/// `MaxPoolArrayLayer.forward_batch`: each channel pooled independently over `kernel_size`-square
/// windows. Returns `(A, argmax)`, both `(N, C*out_height*out_width)` channel-major. `argmax`
/// holds each window's winning slot, numbered row-major `(pr, pc)`, as an `f64`: the values are
/// small exact integers, and it avoids a separate integer array type. The scan uses strict `>`,
/// so the first maximal slot wins, as with `np.argmax` and `PoolUnit`.
#[pyfunction]
pub fn max_pool_forward_batch(x: &RustArray, geometry: &ConvGeometry) -> PyResult<(RustArray, RustArray)> {
    let g = geometry;
    let n = require_matrix(x, None, g.input_size, "max_pool_forward_batch X")?;
    let k = g.kernel_size;
    let size = g.input_channels * g.positions;

    let mut a = vec![0.0; n * size];
    let mut argmax = vec![0.0; n * size];
    for example in 0..n {
        let input = &x.data[example * g.input_size..(example + 1) * g.input_size];
        for c in 0..g.input_channels {
            for out_row in 0..g.out_height {
                for out_col in 0..g.out_width {
                    let mut best_slot = 0;
                    let mut best_value = input[g.input_index(c, out_row, out_col, 0, 0)];
                    for slot in 1..k * k {
                        let value = input[g.input_index(c, out_row, out_col, slot / k, slot % k)];
                        if value > best_value {
                            best_value = value;
                            best_slot = slot;
                        }
                    }
                    let out = example * size + c * g.positions + out_row * g.out_width + out_col;
                    a[out] = best_value;
                    argmax[out] = best_slot as f64;
                }
            }
        }
    }
    Ok((RustArray::from_matrix(a, n, size), RustArray::from_matrix(argmax, n, size)))
}

/// `MaxPoolArrayLayer._downstream`: each window's delta goes to its winning input by scatter-add,
/// so an input that wins several overlapping windows receives all their deltas. The loop runs
/// slot outside window, so each input accumulates in the same order as numpy's one-slice-add-
/// per-slot loop; no matmul is involved, so the result is bit-identical to numpy's.
#[pyfunction]
pub fn max_pool_downstream_batch(
    delta_batch: &RustArray,
    argmax: &RustArray,
    geometry: &ConvGeometry,
) -> PyResult<RustArray> {
    let g = geometry;
    let k = g.kernel_size;
    let size = g.input_channels * g.positions;
    let n = require_matrix(delta_batch, None, size, "max_pool_downstream_batch delta_batch")?;
    require_matrix(argmax, Some(n), size, "max_pool_downstream_batch argmax")?;
    let slot_count = (k * k) as f64;
    let is_slot = |v: f64| v >= 0.0 && v < slot_count && v.fract() == 0.0;
    if let Some(bad) = argmax.data.iter().find(|&&v| !is_slot(v)) {
        return Err(PyValueError::new_err(format!(
            "max_pool_downstream_batch: argmax entries must be slot indices in 0..{}, got {bad}",
            k * k
        )));
    }

    let mut dx = vec![0.0; n * g.input_size];
    for example in 0..n {
        let out = &mut dx[example * g.input_size..(example + 1) * g.input_size];
        for slot in 0..k * k {
            for c in 0..g.input_channels {
                for out_row in 0..g.out_height {
                    for out_col in 0..g.out_width {
                        let window = example * size + c * g.positions + out_row * g.out_width + out_col;
                        if argmax.data[window] as usize == slot {
                            let input = g.input_index(c, out_row, out_col, slot / k, slot % k);
                            out[input] += delta_batch.data[window];
                        }
                    }
                }
            }
        }
    }
    Ok(RustArray::from_matrix(dx, n, g.input_size))
}
