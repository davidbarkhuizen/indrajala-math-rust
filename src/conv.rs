//! The bindings of the core's `conv` (`indrajala_math::conv`): each `#[pyfunction]` lends its
//! arrays to the core's and moves the results in.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::linalg::py_err;
#[allow(unused_imports)]
pub(crate) use indrajala_math::conv::*;

/// The shape arithmetic for one conv or pool layer (`indrajala_math::conv::ConvGeometry`), built
/// once by the Python layer and passed to every call.
#[pyclass(frozen, skip_from_py_object, name = "ConvGeometry")]
#[derive(Clone, Copy, Debug)]
pub struct ConvGeometry(pub indrajala_math::conv::ConvGeometry);

impl std::ops::Deref for ConvGeometry {
    type Target = indrajala_math::conv::ConvGeometry;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
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
        indrajala_math::conv::ConvGeometry::new(input_height, input_width, input_channels, kernel_size, stride)
            .map(ConvGeometry)
            .map_err(py_err)
    }

    #[getter]
    fn input_height(&self) -> usize {
        self.0.input_height
    }

    #[getter]
    fn input_width(&self) -> usize {
        self.0.input_width
    }

    #[getter]
    fn input_channels(&self) -> usize {
        self.0.input_channels
    }

    #[getter]
    fn kernel_size(&self) -> usize {
        self.0.kernel_size
    }

    #[getter]
    fn stride(&self) -> usize {
        self.0.stride
    }

    #[getter]
    fn out_height(&self) -> usize {
        self.0.out_height
    }

    #[getter]
    fn out_width(&self) -> usize {
        self.0.out_width
    }

    #[getter]
    fn positions(&self) -> usize {
        self.0.positions
    }

    #[getter]
    fn fan_in(&self) -> usize {
        self.0.fan_in
    }

    #[getter]
    fn input_size(&self) -> usize {
        self.0.input_size
    }

    fn __repr__(&self) -> String {
        format!(
            "ConvGeometry(input_height={}, input_width={}, input_channels={}, kernel_size={}, stride={})",
            self.input_height, self.input_width, self.input_channels, self.kernel_size, self.stride
        )
    }
}

/// `ConvArrayLayer.forward_batch`, one example at a time: im2col appended to `cols`, the example's
/// `P` rows of `cols @ W.T` with the tiled kernel into one `(P, O)` buffer reused for every
/// example (so each output has `matmul`'s bits), then `A`'s channel-major row appended in order,
/// adding `b` and applying the ReLU as it goes. Returns `(A, cols)`, `A` a vector when `X` is one,
/// `cols` always `(N*P, C*k*k)`: `cols` is kept by the caller for `conv_accumulate_gradient_batch`,
/// as `ConvArrayLayer._cols` is. The pre-activation `Z` is never stored: nothing in the backward
/// pass reads it (`array_relu_mask` masks on `A`).
///
/// Per example, not over the whole batch:
/// at N = 32 the whole-batch `cols`, product and `A` are 1.4-1.6 MB each, past L2, and zeroing
/// them and then scattering into `A` made the op cost 1.5-2.8x its N single-example calls. Here
/// only `cols` and `A` are batch-sized, and each is written once, in order; the example's slab
/// of `cols` and its 43 KB product are still in cache when they are read. The product runs on
/// one thread: threading the whole batch's product over its rows made no difference at N = 512.
#[pyfunction]
pub fn conv_forward_batch(
    w: &RustArray,
    x: &RustArray,
    b: &RustArray,
    geometry: &ConvGeometry,
) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::conv::conv_forward_batch(w, x, b, geometry)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

/// `LinearConvArrayLayer.forward_batch`: `conv_forward_batch`'s products without the bias and the
/// ReLU, the pre-activations a batch-norm layer normalizes. Returns `(A, cols)` as that op does.
///
/// Its own copy of `conv_forward_batch`'s body, not a shared helper: with the body shared, the
/// conv mini-batch op profile put `conv_forward_batch` 8-17% above its unshared form.
#[pyfunction]
pub fn conv_linear_forward_batch(
    w: &RustArray,
    x: &RustArray,
    geometry: &ConvGeometry,
) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::conv::conv_linear_forward_batch(w, x, geometry)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

/// `ConvArrayLayer._downstream`: per example, `dcols = D @ W`, `D` the example's deltas regrouped to
/// `(P, O)`, then col2im as a scatter-add. The loop runs kernel offset `(kr, kc)` outside output
/// position, so each input accumulates its contributions in the same order as numpy's
/// one-slice-add-per-offset loop. `dX` is a vector when `delta_batch` is one.
///
/// Per example, as `conv_forward_batch`: each example's `(P, C*k*k)` product goes into one buffer
/// reused for every example, still in cache when col2im reads it. The whole batch's product
/// (10.6 MB at 26x26x8, N = 32), threaded over its rows, was memory-bound on the Pi: with the
/// NEON kernel 4 threads ran no faster than one, and the op took 15 ms against the scalar kernel's
/// 9.5 (the aarch64 workplan's stage 7). Threads take whole examples, each with its own buffers
/// and `dX` rows, so no output's value depends on the split.
#[pyfunction]
pub fn conv_downstream_batch(w: &RustArray, delta_batch: &RustArray, geometry: &ConvGeometry) -> PyResult<RustArray> {
    indrajala_math::conv::conv_downstream_batch(w, delta_batch, geometry)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `ConvArrayLayer._accumulate`: `grad_W += D @ cols` with `matmul_long_k`, `D` the deltas regrouped to
/// `(O, N*P)`, and `grad_b += ` each channel's delta sum over every example and position. Returns
/// the updated pair for the caller to rebind, as `layer_accumulate_gradient_batch` does. A vector
/// `delta_batch` is one example, whose `cols` has `P` rows.
#[pyfunction]
pub fn conv_accumulate_gradient_batch(
    delta_batch: &RustArray,
    cols: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    geometry: &ConvGeometry,
) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::conv::conv_accumulate_gradient_batch(delta_batch, cols, grad_w, grad_b, geometry)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

/// `LinearConvArrayLayer.accumulate_gradient_batch`: `conv_accumulate_gradient_batch`'s `grad_W`,
/// without its `grad_b`. Returns the updated `grad_W`.
#[pyfunction]
pub fn conv_linear_accumulate_gradient_batch(
    delta_batch: &RustArray,
    cols: &RustArray,
    grad_w: &RustArray,
    geometry: &ConvGeometry,
) -> PyResult<RustArray> {
    indrajala_math::conv::conv_linear_accumulate_gradient_batch(delta_batch, cols, grad_w, geometry)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `MaxPoolArrayLayer.forward_batch`: each channel pooled independently over `kernel_size`-square
/// windows. Returns `(A, argmax)`, both `(N, C*out_height*out_width)` channel-major, or both
/// vectors when `X` is one. `argmax`
/// holds each window's winning slot, numbered row-major `(pr, pc)`, as an `f64`: the values are
/// small exact integers, and it avoids a separate integer array type. The scan uses strict `>`,
/// so the first maximal slot wins, as with `np.argmax` and `PoolUnit`. Both kernels scan in that
/// order; 2x2 windows at stride 2 take a fixed-size one, which runs in about a third of the time.
#[pyfunction]
pub fn max_pool_forward_batch(x: &RustArray, geometry: &ConvGeometry) -> PyResult<(RustArray, RustArray)> {
    indrajala_math::conv::max_pool_forward_batch(x, geometry)
        .map(|r| {
            let (r0, r1) = r;
            (RustArray::from(r0), RustArray::from(r1))
        })
        .map_err(py_err)
}

/// `MaxPoolArrayLayer._downstream`: each window's delta goes to its winning input by scatter-add,
/// so an input that wins several overlapping windows receives all their deltas. With overlapping
/// windows (stride < `kernel_size`) the loop runs slot outside window, so each input accumulates
/// in the same order as numpy's one-slice-add-per-slot loop; no matmul is involved, so the result
/// is bit-identical to numpy's. Without overlap each input receives at most one delta, so one pass
/// in window order stores `0.0 + d`, the bits the add into the zeroed `dX` gives (`-0.0` included).
/// `delta_batch` and `argmax` must have the same rank; `dX` has it too.
#[pyfunction]
pub fn max_pool_downstream_batch(
    delta_batch: &RustArray,
    argmax: &RustArray,
    geometry: &ConvGeometry,
) -> PyResult<RustArray> {
    indrajala_math::conv::max_pool_downstream_batch(delta_batch, argmax, geometry)
        .map(RustArray::from)
        .map_err(py_err)
}
