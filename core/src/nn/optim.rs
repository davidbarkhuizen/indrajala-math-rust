//! The optimizers' parameter updates, one call per `ArrayLayer` update method: plain SGD (a
//! minibatch's accumulated gradient, or one example's step), momentum, L2 and Adam.

use crate::array::{Array, Shape};
use crate::num::scalar::powi;
use crate::ops::arith::{require_same_shape, same_shape_elementwise};
use crate::Error;

fn require_batch_size(batch_size: usize, context: &str) -> Result<(), Error> {
    if batch_size == 0 {
        return Err(Error::invalid(format!("{context} requires batch_size >= 1")));
    }
    Ok(())
}

/// `ArrayLayer.apply_accumulated_gradient`: minibatch SGD in Goyal et al. 2017's grouping, eq. (2),
/// the mean gradient first and then the rate: `self.W -= learning_rate * (self._grad_W /
/// batch_size); self.b -= learning_rate * (self._grad_b / batch_size)` - shape-agnostic
/// (`W`/`grad_W` are always the same shape as each other, likewise `b`/`grad_b`), so one function
/// covers both the single-example (`batch_size=1`) and batched caller.
pub fn apply_accumulated_gradient(
    w: &Array,
    b: &Array,
    grad_w: &Array,
    grad_b: &Array,
    learning_rate: f64,
    batch_size: usize,
) -> Result<(Array, Array), Error> {
    require_same_shape(w, grad_w, "layer_apply_accumulated_gradient (W, grad_W)")?;
    require_same_shape(b, grad_b, "layer_apply_accumulated_gradient (b, grad_b)")?;
    require_batch_size(batch_size, "layer_apply_accumulated_gradient")?;
    let batch_size = batch_size as f64;
    let new_w = Array {
        data: same_shape_elementwise(&w.data, &grad_w.data, |wv, gv| wv - learning_rate * (gv / batch_size)),
        shape: w.shape,
    };
    let new_b = Array {
        data: same_shape_elementwise(&b.data, &grad_b.data, |bv, gv| bv - learning_rate * (gv / batch_size)),
        shape: b.shape,
    };
    Ok((new_w, new_b))
}

/// One single-example plain-SGD step: `dense::accumulate_gradient` into fresh zero accumulators,
/// then `apply_accumulated_gradient` with `batch_size=1`, in one pass and one allocation
/// per parameter. `RustArrayNetworkBase.learn` runs exactly that pair on every layer, and apply
/// resets the accumulators, so at batch size 1 they are always fresh. Computes `w - lr * (0.0 +
/// delta[i] * x[j])`: the `0.0 +` is the add into the zero accumulator, kept on purpose. It turns
/// a `-0.0` product into `+0.0`, which changes the result where `w` is `-0.0`, so dropping it
/// would break bit-identity with the pair it replaces.
pub fn sgd_step(
    w: &Array,
    b: &Array,
    delta: &Array,
    input_activation: &Array,
    learning_rate: f64,
) -> Result<(Array, Array), Error> {
    let (m, n) = match (delta.shape, input_activation.shape) {
        (Shape::Vector(m), Shape::Vector(n)) => (m, n),
        (d_shape, x_shape) => {
            return Err(Error::invalid(format!(
                "layer_sgd_step requires 1D delta and input_activation, got shapes {:?} and {:?}",
                d_shape, x_shape
            )))
        }
    };
    if w.shape != Shape::Matrix(m, n) || b.shape != Shape::Vector(m) {
        return Err(Error::invalid(format!(
            "layer_sgd_step requires W of shape {:?} and b of shape {:?}, got {:?} and {:?}",
            Shape::Matrix(m, n),
            Shape::Vector(m),
            w.shape,
            b.shape
        )));
    }
    // apply_accumulated_gradient's learning_rate * (g / batch_size) is exactly
    // learning_rate * g at batch_size=1
    let scale = learning_rate;
    let x = &input_activation.data;
    let mut w_data = Vec::with_capacity(m * n);
    for (w_row, &d) in w.data.chunks_exact(n.max(1)).zip(delta.data.iter()) {
        w_data.extend(w_row.iter().zip(x.iter()).map(|(&wv, &xj)| wv - scale * (0.0 + d * xj)));
    }
    let b_data = b
        .data
        .iter()
        .zip(delta.data.iter())
        .map(|(&bv, &d)| bv - scale * (0.0 + d))
        .collect();
    Ok((Array::from_matrix(w_data, m, n), Array::from_vector(b_data)))
}

/// `AdamArrayLayer.apply_accumulated_gradient`: the Adam (Kingma & Ba, 2014) update rule, as one
/// fused call per parameter (`W` or `b`) instead of composing it from several `Array` operators.
/// Shape-agnostic like `apply_accumulated_gradient`
/// above, so this one helper covers both the `W`/`grad_W`/`m_W`/`v_W` (2D) and `b`/`grad_b`/`m_b`/
/// `v_b` (1D) cases. `t` is the step count *after* incrementing - mirrors
/// `AdamArrayLayer._t += 1` happening before the bias-correction terms are computed, so the
/// Python caller increments its own `_t` and passes the new value in rather than this function
/// owning the counter.
#[allow(clippy::too_many_arguments)]
fn adam_update(
    param: &Array,
    grad: &Array,
    m: &Array,
    v: &Array,
    t: u32,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    learning_rate: f64,
    batch_size: f64,
    weight_decay: f64,
) -> (Array, Array, Array) {
    // decoupled weight decay (AdamW, Loshchilov & Hutter 2019), in PyTorch's AdamW grouping: the
    // parameter shrinks by learning_rate * weight_decay before the Adam step is taken from it. At
    // weight_decay 0 the factor is exactly 1.0, so the step is Adam's, bit for bit
    let decay = 1.0 - learning_rate * weight_decay;
    let bias_correction1 = 1.0 - powi(beta1, t);
    let bias_correction2 = 1.0 - powi(beta2, t);

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
        new_param_data.push(param.data[i] * decay - learning_rate * m_hat / (v_hat.sqrt() + epsilon));
        new_m_data.push(new_m);
        new_v_data.push(new_v);
    }

    (
        Array {
            data: new_param_data,
            shape: param.shape,
        },
        Array {
            data: new_m_data,
            shape: param.shape,
        },
        Array {
            data: new_v_data,
            shape: param.shape,
        },
    )
}

/// `AdamArrayLayer.apply_accumulated_gradient`: the Adam (Kingma & Ba, 2014) update of `W` and `b`
/// from their accumulated gradients over `batch_size` examples, with step count `t` (after
/// incrementing, so 1 or more), and AdamW's decoupled weight decay `weight_decay` on `W` only (`b`
/// isn't decayed; 0 is Adam): `W * (1 - learning_rate * weight_decay) - learning_rate * m_hat /
/// (sqrt(v_hat) + epsilon)`. Returns `(W, b, m_W, v_W, m_b, v_b)`, the new parameters and moments.
#[allow(clippy::too_many_arguments)]
pub fn adam_apply_accumulated_gradient(
    w: &Array,
    b: &Array,
    grad_w: &Array,
    grad_b: &Array,
    m_w: &Array,
    v_w: &Array,
    m_b: &Array,
    v_b: &Array,
    t: u32,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    learning_rate: f64,
    batch_size: usize,
    weight_decay: f64,
) -> Result<(Array, Array, Array, Array, Array, Array), Error> {
    require_same_shape(w, grad_w, "layer_adam_apply_accumulated_gradient (W, grad_W)")?;
    require_same_shape(w, m_w, "layer_adam_apply_accumulated_gradient (W, m_W)")?;
    require_same_shape(w, v_w, "layer_adam_apply_accumulated_gradient (W, v_W)")?;
    require_same_shape(b, grad_b, "layer_adam_apply_accumulated_gradient (b, grad_b)")?;
    require_same_shape(b, m_b, "layer_adam_apply_accumulated_gradient (b, m_b)")?;
    require_same_shape(b, v_b, "layer_adam_apply_accumulated_gradient (b, v_b)")?;
    require_batch_size(batch_size, "layer_adam_apply_accumulated_gradient")?;
    if t == 0 {
        return Err(Error::invalid(
            "layer_adam_apply_accumulated_gradient requires t >= 1 (the step count after incrementing)",
        ));
    }
    if !(weight_decay.is_finite() && weight_decay >= 0.0) {
        return Err(Error::invalid(format!(
            "layer_adam_apply_accumulated_gradient requires a finite weight_decay >= 0; got {weight_decay}"
        )));
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
        weight_decay,
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
        0.0,
    );

    Ok((new_w, new_b, new_m_w, new_v_w, new_m_b, new_v_b))
}

/// `L2ArrayLayer.apply_accumulated_gradient`: weight decay as Goyal et al. 2017's eq. (8), `λW`
/// added to the mean gradient: `W -= learning_rate * (grad_W / batch_size + l2_lambda * W);
/// b -= learning_rate * (grad_b / batch_size)` (bias unregularized). No persistent
/// per-parameter state at all (unlike
/// `adam_apply_accumulated_gradient`/`momentum_apply_accumulated_gradient`), so this
/// takes only `W`/`b`/`grad_W`/`grad_b` plus the scalar `l2_lambda` - the simplest fused op in
/// this round.
pub fn l2_apply_accumulated_gradient(
    w: &Array,
    b: &Array,
    grad_w: &Array,
    grad_b: &Array,
    l2_lambda: f64,
    learning_rate: f64,
    batch_size: usize,
) -> Result<(Array, Array), Error> {
    require_same_shape(w, grad_w, "layer_l2_apply_accumulated_gradient (W, grad_W)")?;
    require_same_shape(b, grad_b, "layer_l2_apply_accumulated_gradient (b, grad_b)")?;
    require_batch_size(batch_size, "layer_l2_apply_accumulated_gradient")?;
    let batch_size = batch_size as f64;
    let new_w = Array {
        data: same_shape_elementwise(&w.data, &grad_w.data, |wv, gv| {
            wv - learning_rate * (gv / batch_size + l2_lambda * wv)
        }),
        shape: w.shape,
    };
    let new_b = Array {
        data: same_shape_elementwise(&b.data, &grad_b.data, |bv, gv| bv - learning_rate * (gv / batch_size)),
        shape: b.shape,
    };
    Ok((new_w, new_b))
}

/// `MomentumArrayLayer.apply_accumulated_gradient`: momentum as Goyal et al. 2017's eq. (9),
/// `velocity = momentum * velocity + grad / batch_size; param -= learning_rate * velocity`. It
/// replaces Rumelhart et al.'s eq. (10), which folds the rate into the velocity and so needs a
/// correction when the rate changes. One velocity array per parameter tensor, shape-agnostic like
/// `apply_accumulated_gradient` (covers both the `W`/`grad_W`/`velocity_W` (2D) and
/// `b`/`grad_b`/`velocity_b` (1D) cases via two calls from the Python caller, the same convention
/// `apply_accumulated_gradient` itself uses).
#[allow(clippy::too_many_arguments)]
pub fn momentum_apply_accumulated_gradient(
    w: &Array,
    b: &Array,
    grad_w: &Array,
    grad_b: &Array,
    velocity_w: &Array,
    velocity_b: &Array,
    momentum: f64,
    learning_rate: f64,
    batch_size: usize,
) -> Result<(Array, Array, Array, Array), Error> {
    require_same_shape(w, grad_w, "layer_momentum_apply_accumulated_gradient (W, grad_W)")?;
    require_same_shape(
        w,
        velocity_w,
        "layer_momentum_apply_accumulated_gradient (W, velocity_W)",
    )?;
    require_same_shape(b, grad_b, "layer_momentum_apply_accumulated_gradient (b, grad_b)")?;
    require_same_shape(
        b,
        velocity_b,
        "layer_momentum_apply_accumulated_gradient (b, velocity_b)",
    )?;
    require_batch_size(batch_size, "layer_momentum_apply_accumulated_gradient")?;
    let batch_size = batch_size as f64;

    let new_velocity_w_data: Vec<f64> = (0..w.data.len())
        .map(|i| momentum * velocity_w.data[i] + grad_w.data[i] / batch_size)
        .collect();
    let new_w = Array {
        data: same_shape_elementwise(&w.data, &new_velocity_w_data, |wv, uv| wv - learning_rate * uv),
        shape: w.shape,
    };
    let new_velocity_w = Array {
        data: new_velocity_w_data,
        shape: w.shape,
    };

    let new_velocity_b_data: Vec<f64> = (0..b.data.len())
        .map(|i| momentum * velocity_b.data[i] + grad_b.data[i] / batch_size)
        .collect();
    let new_b = Array {
        data: same_shape_elementwise(&b.data, &new_velocity_b_data, |bv, uv| bv - learning_rate * uv),
        shape: b.shape,
    };
    let new_velocity_b = Array {
        data: new_velocity_b_data,
        shape: b.shape,
    };

    Ok((new_w, new_b, new_velocity_w, new_velocity_b))
}
