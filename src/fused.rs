use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::RustArray;
use crate::linalg::{matmul, outer};
use crate::ops::same_shape_elementwise;
use crate::ufuncs::{array_softmax, sum_axis0};

/// docs/rust-production-cutover.md's phase 0b: one Rust function per `ArrayLayer` method,
/// doing the entire computation in one call instead of composing it from several separate
/// `Array` operator/ufunc calls in Python - each of those crosses the Python/Rust boundary and
/// allocates a new `Array`, which is cause 1 of that document's "the decisive finding" (call
/// count, not per-call cost, dominates at this crate's naive-matmul stage). Every function here
/// mirrors one `indrajala_ml/model/array_layer.py` method's formula exactly - see that file for
/// the reference this crate is being checked against.

fn require_same_shape(a: &RustArray, b: &RustArray, context: &str) -> PyResult<()> {
    if a.shape != b.shape {
        return Err(PyValueError::new_err(format!(
            "{context} requires matching shapes, got {:?} and {:?}",
            a.shape, b.shape
        )));
    }
    Ok(())
}

fn sigmoid(z: &RustArray) -> RustArray {
    RustArray {
        data: z.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
        shape: z.shape,
    }
}

/// `ArrayLayer.forward`: `sigmoid(self.W @ x + self.b)`, `x`/`b` both 1D.
#[pyfunction]
pub fn layer_forward(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    let z = matmul(w, x)?;
    let z = z.combine_with_array(b, |a, bv| a + bv, "add")?;
    Ok(sigmoid(&z))
}

/// `ArrayLayer.forward_batch`: `sigmoid(X @ self.W.T + self.b)`, `X` 2D (`batch, input_size`).
#[pyfunction]
pub fn layer_forward_batch(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    let w_t = w.transpose();
    let z = matmul(x, &w_t)?;
    let z = z.combine_with_array(b, |a, bv| a + bv, "add")?;
    Ok(sigmoid(&z))
}

/// `ArrayLayer.compute_output_delta`/`compute_output_delta_batch`: `(a - reference) * a * (1 -
/// a)` - one elementwise formula, shape-agnostic (works for both the single-example 1D case and
/// the batched 2D case), so unlike `forward`/`forward_batch` this needs only one function.
#[pyfunction]
pub fn layer_output_delta(a: &RustArray, reference: &RustArray) -> PyResult<RustArray> {
    require_same_shape(a, reference, "layer_output_delta")?;
    let data = a
        .data
        .iter()
        .zip(reference.data.iter())
        .map(|(&av, &rv)| (av - rv) * av * (1.0 - av))
        .collect();
    Ok(RustArray {
        data,
        shape: a.shape,
    })
}

/// `ArrayLayer.compute_hidden_delta`: `(next_layer.W.T @ next_layer.delta) * self.a * (1 -
/// self.a)`, single-example (`next_delta`/`a` both 1D).
#[pyfunction]
pub fn layer_hidden_delta(
    next_w: &RustArray,
    next_delta: &RustArray,
    a: &RustArray,
) -> PyResult<RustArray> {
    let downstream = matmul(&next_w.transpose(), next_delta)?;
    require_same_shape(&downstream, a, "layer_hidden_delta")?;
    let data = downstream
        .data
        .iter()
        .zip(a.data.iter())
        .map(|(&d, &av)| d * av * (1.0 - av))
        .collect();
    Ok(RustArray {
        data,
        shape: a.shape,
    })
}

/// `ArrayLayer.compute_hidden_delta_batch`: `(next_layer.delta_batch @ next_layer.W) * self.A *
/// (1 - self.A)` - batched, no transpose on `next_w` (unlike the single-example case above,
/// since `next_delta_batch`'s batch axis is on the left instead of `next_w`'s being on the
/// left), so this is a genuinely different call shape, not just a shape-agnostic reuse of
/// `layer_hidden_delta`.
#[pyfunction]
pub fn layer_hidden_delta_batch(
    next_w: &RustArray,
    next_delta_batch: &RustArray,
    a_batch: &RustArray,
) -> PyResult<RustArray> {
    let downstream = matmul(next_delta_batch, next_w)?;
    require_same_shape(&downstream, a_batch, "layer_hidden_delta_batch")?;
    let data = downstream
        .data
        .iter()
        .zip(a_batch.data.iter())
        .map(|(&d, &av)| d * av * (1.0 - av))
        .collect();
    Ok(RustArray {
        data,
        shape: a_batch.shape,
    })
}

/// `ArrayLayer.accumulate_gradient`: `self._grad_W += outer(delta, input_activation); self._grad_b
/// += delta`, single-example. Returns the updated `(grad_W, grad_b)` pair rather than mutating in
/// place - `Array` is immutable from Python's own operators (`+=` rebinds via `__iadd__`, which
/// this crate already has), so the caller rebinds `self._grad_W`/`self._grad_b` the same way.
#[pyfunction]
pub fn layer_accumulate_gradient(
    delta: &RustArray,
    input_activation: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
) -> PyResult<(RustArray, RustArray)> {
    let outer_product = outer(delta, input_activation)?;
    let new_grad_w = grad_w.combine_with_array(&outer_product, |g, o| g + o, "add")?;
    let new_grad_b = grad_b.combine_with_array(delta, |g, d| g + d, "add")?;
    Ok((new_grad_w, new_grad_b))
}

/// `ArrayLayer.accumulate_gradient_batch`: `self._grad_W += self.delta_batch.T @
/// input_activation_batch; self._grad_b += self.delta_batch.sum(axis=0)`.
#[pyfunction]
pub fn layer_accumulate_gradient_batch(
    delta_batch: &RustArray,
    input_activation_batch: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
) -> PyResult<(RustArray, RustArray)> {
    let grad_w_update = matmul(&delta_batch.transpose(), input_activation_batch)?;
    let new_grad_w = grad_w.combine_with_array(&grad_w_update, |g, u| g + u, "add")?;
    let grad_b_update = sum_axis0(delta_batch)?;
    let new_grad_b = grad_b.combine_with_array(&grad_b_update, |g, u| g + u, "add")?;
    Ok((new_grad_w, new_grad_b))
}

/// `ArrayLayer.apply_accumulated_gradient`: `self.W -= learning_rate * self._grad_W /
/// batch_size; self.b -= learning_rate * self._grad_b / batch_size` - shape-agnostic (`W`/`grad_W`
/// are always the same shape as each other, likewise `b`/`grad_b`), so one function covers both
/// the single-example (`batch_size=1`) and batched caller.
#[pyfunction]
pub fn layer_apply_accumulated_gradient(
    w: &RustArray,
    b: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    learning_rate: f64,
    batch_size: usize,
) -> PyResult<(RustArray, RustArray)> {
    require_same_shape(w, grad_w, "layer_apply_accumulated_gradient (W, grad_W)")?;
    require_same_shape(b, grad_b, "layer_apply_accumulated_gradient (b, grad_b)")?;
    if batch_size == 0 {
        return Err(PyValueError::new_err(
            "layer_apply_accumulated_gradient requires batch_size >= 1",
        ));
    }
    let scale = learning_rate / (batch_size as f64);
    let new_w = RustArray {
        data: same_shape_elementwise(&w.data, &grad_w.data, |wv, gv| wv - scale * gv),
        shape: w.shape,
    };
    let new_b = RustArray {
        data: same_shape_elementwise(&b.data, &grad_b.data, |bv, gv| bv - scale * gv),
        shape: b.shape,
    };
    Ok((new_w, new_b))
}

/// `AdamArrayLayer.apply_accumulated_gradient`: the Adam (Kingma & Ba, 2014) update rule - see
/// docs/adam-array-layer.md - as one fused call per parameter (`W` or `b`) instead of composing
/// it from several `Array` operators. Shape-agnostic like `layer_apply_accumulated_gradient`
/// above, so this one helper covers both the `W`/`grad_W`/`m_W`/`v_W` (2D) and `b`/`grad_b`/`m_b`/
/// `v_b` (1D) cases. `t` is the step count *after* incrementing - mirrors
/// `AdamArrayLayer._t += 1` happening before the bias-correction terms are computed, so the
/// Python caller increments its own `_t` and passes the new value in rather than this function
/// owning the counter.
#[allow(clippy::too_many_arguments)]
fn adam_update(
    param: &RustArray,
    grad: &RustArray,
    m: &RustArray,
    v: &RustArray,
    t: u32,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    learning_rate: f64,
    batch_size: f64,
) -> (RustArray, RustArray, RustArray) {
    let bias_correction1 = 1.0 - beta1.powi(t as i32);
    let bias_correction2 = 1.0 - beta2.powi(t as i32);

    let n = param.data.len();
    let mut new_param_data = Vec::with_capacity(n);
    let mut new_m_data = Vec::with_capacity(n);
    let mut new_v_data = Vec::with_capacity(n);

    for i in 0..n {
        let g = grad.data[i] / batch_size;
        let new_m = beta1 * m.data[i] + (1.0 - beta1) * g;
        let new_v = beta2 * v.data[i] + (1.0 - beta2) * g * g;
        let m_hat = new_m / bias_correction1;
        let v_hat = new_v / bias_correction2;
        new_param_data.push(param.data[i] - learning_rate * m_hat / (v_hat.sqrt() + epsilon));
        new_m_data.push(new_m);
        new_v_data.push(new_v);
    }

    (
        RustArray {
            data: new_param_data,
            shape: param.shape,
        },
        RustArray {
            data: new_m_data,
            shape: param.shape,
        },
        RustArray {
            data: new_v_data,
            shape: param.shape,
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn layer_adam_apply_accumulated_gradient(
    w: &RustArray,
    b: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    m_w: &RustArray,
    v_w: &RustArray,
    m_b: &RustArray,
    v_b: &RustArray,
    t: u32,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    learning_rate: f64,
    batch_size: usize,
) -> PyResult<(RustArray, RustArray, RustArray, RustArray, RustArray, RustArray)> {
    require_same_shape(w, grad_w, "layer_adam_apply_accumulated_gradient (W, grad_W)")?;
    require_same_shape(w, m_w, "layer_adam_apply_accumulated_gradient (W, m_W)")?;
    require_same_shape(w, v_w, "layer_adam_apply_accumulated_gradient (W, v_W)")?;
    require_same_shape(b, grad_b, "layer_adam_apply_accumulated_gradient (b, grad_b)")?;
    require_same_shape(b, m_b, "layer_adam_apply_accumulated_gradient (b, m_b)")?;
    require_same_shape(b, v_b, "layer_adam_apply_accumulated_gradient (b, v_b)")?;
    if batch_size == 0 {
        return Err(PyValueError::new_err(
            "layer_adam_apply_accumulated_gradient requires batch_size >= 1",
        ));
    }
    if t == 0 {
        return Err(PyValueError::new_err(
            "layer_adam_apply_accumulated_gradient requires t >= 1 (the step count after incrementing)",
        ));
    }

    let (new_w, new_m_w, new_v_w) = adam_update(
        w,
        grad_w,
        m_w,
        v_w,
        t,
        beta1,
        beta2,
        epsilon,
        learning_rate,
        batch_size as f64,
    );
    let (new_b, new_m_b, new_v_b) = adam_update(
        b,
        grad_b,
        m_b,
        v_b,
        t,
        beta1,
        beta2,
        epsilon,
        learning_rate,
        batch_size as f64,
    );

    Ok((new_w, new_b, new_m_w, new_v_w, new_m_b, new_v_b))
}

/// `L2ArrayLayer.apply_accumulated_gradient`: `W -= learning_rate * (grad_W / batch_size +
/// l2_lambda * W); b -= learning_rate * grad_b / batch_size` (bias unregularized) - see
/// docs/l2-array-layer.md. No persistent per-parameter state at all (unlike
/// `layer_adam_apply_accumulated_gradient`/`layer_momentum_apply_accumulated_gradient`), so this
/// takes only `W`/`b`/`grad_W`/`grad_b` plus the scalar `l2_lambda` - the simplest fused op in
/// this round.
#[pyfunction]
pub fn layer_l2_apply_accumulated_gradient(
    w: &RustArray,
    b: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    l2_lambda: f64,
    learning_rate: f64,
    batch_size: usize,
) -> PyResult<(RustArray, RustArray)> {
    require_same_shape(w, grad_w, "layer_l2_apply_accumulated_gradient (W, grad_W)")?;
    require_same_shape(b, grad_b, "layer_l2_apply_accumulated_gradient (b, grad_b)")?;
    if batch_size == 0 {
        return Err(PyValueError::new_err(
            "layer_l2_apply_accumulated_gradient requires batch_size >= 1",
        ));
    }
    let scale = learning_rate / (batch_size as f64);
    let new_w = RustArray {
        data: same_shape_elementwise(&w.data, &grad_w.data, |wv, gv| {
            wv - scale * gv - learning_rate * l2_lambda * wv
        }),
        shape: w.shape,
    };
    let new_b = RustArray {
        data: same_shape_elementwise(&b.data, &grad_b.data, |bv, gv| bv - scale * gv),
        shape: b.shape,
    };
    Ok((new_w, new_b))
}

/// `MomentumArrayLayer.apply_accumulated_gradient`: `delta = learning_rate * grad / batch_size +
/// momentum * prev_delta; param -= delta` - see docs/momentum-array-layer.md. One previous-delta
/// array per parameter tensor, shape-agnostic like `layer_apply_accumulated_gradient` (covers
/// both the `W`/`grad_W`/`prev_delta_W` (2D) and `b`/`grad_b`/`prev_delta_b` (1D) cases via two
/// calls from the Python caller, the same convention `layer_apply_accumulated_gradient` itself
/// uses).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn layer_momentum_apply_accumulated_gradient(
    w: &RustArray,
    b: &RustArray,
    grad_w: &RustArray,
    grad_b: &RustArray,
    prev_delta_w: &RustArray,
    prev_delta_b: &RustArray,
    momentum: f64,
    learning_rate: f64,
    batch_size: usize,
) -> PyResult<(RustArray, RustArray, RustArray, RustArray)> {
    require_same_shape(w, grad_w, "layer_momentum_apply_accumulated_gradient (W, grad_W)")?;
    require_same_shape(w, prev_delta_w, "layer_momentum_apply_accumulated_gradient (W, prev_delta_W)")?;
    require_same_shape(b, grad_b, "layer_momentum_apply_accumulated_gradient (b, grad_b)")?;
    require_same_shape(b, prev_delta_b, "layer_momentum_apply_accumulated_gradient (b, prev_delta_b)")?;
    if batch_size == 0 {
        return Err(PyValueError::new_err(
            "layer_momentum_apply_accumulated_gradient requires batch_size >= 1",
        ));
    }
    let scale = learning_rate / (batch_size as f64);

    let new_delta_w_data: Vec<f64> = (0..w.data.len())
        .map(|i| scale * grad_w.data[i] + momentum * prev_delta_w.data[i])
        .collect();
    let new_w = RustArray {
        data: same_shape_elementwise(&w.data, &new_delta_w_data, |wv, dv| wv - dv),
        shape: w.shape,
    };
    let new_prev_delta_w = RustArray {
        data: new_delta_w_data,
        shape: w.shape,
    };

    let new_delta_b_data: Vec<f64> = (0..b.data.len())
        .map(|i| scale * grad_b.data[i] + momentum * prev_delta_b.data[i])
        .collect();
    let new_b = RustArray {
        data: same_shape_elementwise(&b.data, &new_delta_b_data, |bv, dv| bv - dv),
        shape: b.shape,
    };
    let new_prev_delta_b = RustArray {
        data: new_delta_b_data,
        shape: b.shape,
    };

    Ok((new_w, new_b, new_prev_delta_w, new_prev_delta_b))
}

/// `ReLUArrayLayer.forward`: `max(0, self.W @ x + self.b)`, `x`/`b` both 1D - see
/// docs/relu-array-layer.md. Inlines `array_relu`'s own formula (`ufuncs.rs`) rather than
/// calling it as a separate op, the same "one Rust call per layer method" discipline every
/// other `fused.rs` function follows.
#[pyfunction]
pub fn layer_relu_forward(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    let z = matmul(w, x)?;
    let z = z.combine_with_array(b, |a, bv| a + bv, "add")?;
    Ok(RustArray {
        data: z.data.iter().map(|&v| v.max(0.0)).collect(),
        shape: z.shape,
    })
}

/// `ReLUArrayLayer.forward_batch`: `max(0, X @ self.W.T + self.b)`, `X` 2D (`batch, input_size`).
#[pyfunction]
pub fn layer_relu_forward_batch(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    let w_t = w.transpose();
    let z = matmul(x, &w_t)?;
    let z = z.combine_with_array(b, |a, bv| a + bv, "add")?;
    Ok(RustArray {
        data: z.data.iter().map(|&v| v.max(0.0)).collect(),
        shape: z.shape,
    })
}

/// `ReLUArrayLayer.compute_hidden_delta`: `(next_layer.W.T @ next_layer.delta) * (self.a > 0.0)` -
/// inlines `array_relu_mask`'s own formula rather than calling it as a separate op, single-example
/// (`next_delta`/`a` both 1D).
#[pyfunction]
pub fn layer_relu_hidden_delta(
    next_w: &RustArray,
    next_delta: &RustArray,
    a: &RustArray,
) -> PyResult<RustArray> {
    let downstream = matmul(&next_w.transpose(), next_delta)?;
    require_same_shape(&downstream, a, "layer_relu_hidden_delta")?;
    let data = downstream
        .data
        .iter()
        .zip(a.data.iter())
        .map(|(&d, &av)| if av > 0.0 { d } else { 0.0 })
        .collect();
    Ok(RustArray {
        data,
        shape: a.shape,
    })
}

/// `ReLUArrayLayer.compute_hidden_delta_batch`: `(next_layer.delta_batch @ next_layer.W) *
/// (self.A > 0.0)` - batched, no transpose on `next_w` (same shape-of-call-sites distinction
/// `layer_hidden_delta_batch` itself already documents).
#[pyfunction]
pub fn layer_relu_hidden_delta_batch(
    next_w: &RustArray,
    next_delta_batch: &RustArray,
    a_batch: &RustArray,
) -> PyResult<RustArray> {
    let downstream = matmul(next_delta_batch, next_w)?;
    require_same_shape(&downstream, a_batch, "layer_relu_hidden_delta_batch")?;
    let data = downstream
        .data
        .iter()
        .zip(a_batch.data.iter())
        .map(|(&d, &av)| if av > 0.0 { d } else { 0.0 })
        .collect();
    Ok(RustArray {
        data,
        shape: a_batch.shape,
    })
}

/// `SoftmaxArrayLayer.forward`: `array_softmax(self.W @ x + self.b)`, `x`/`b` both 1D - see
/// docs/softmax-array-layer.md. Reuses `array_softmax` (`ufuncs.rs`, stage 2's primitive)
/// directly rather than inlining its max/sum reduction, unlike ReLU's trivial elementwise
/// formula - the extra call is Rust-internal, not a second Python/Rust FFI crossing.
#[pyfunction]
pub fn layer_softmax_forward(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    let z = matmul(w, x)?;
    let z = z.combine_with_array(b, |a, bv| a + bv, "add")?;
    Ok(array_softmax(&z))
}

/// `SoftmaxArrayLayer.forward_batch`: `array_softmax(X @ self.W.T + self.b)`, row-wise
/// normalization, `X` 2D (`batch, input_size`).
#[pyfunction]
pub fn layer_softmax_forward_batch(w: &RustArray, x: &RustArray, b: &RustArray) -> PyResult<RustArray> {
    let w_t = w.transpose();
    let z = matmul(x, &w_t)?;
    let z = z.combine_with_array(b, |a, bv| a + bv, "add")?;
    Ok(array_softmax(&z))
}

/// `SoftmaxArrayLayer.compute_output_delta`/`compute_output_delta_batch`: `a - reference` -
/// `SoftmaxOutputNode`'s own simplification, no `a*(1-a)` damping term at all - shape-agnostic
/// like `layer_output_delta`, so one function covers both the single-example and batched case.
#[pyfunction]
pub fn layer_softmax_output_delta(a: &RustArray, reference: &RustArray) -> PyResult<RustArray> {
    require_same_shape(a, reference, "layer_softmax_output_delta")?;
    Ok(RustArray {
        data: same_shape_elementwise(&a.data, &reference.data, |av, rv| av - rv),
        shape: a.shape,
    })
}
