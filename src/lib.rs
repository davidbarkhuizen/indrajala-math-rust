use pyo3::prelude::*;

mod array;
mod fused;
mod linalg;
mod mnist;
mod ops;
mod random;
mod ufuncs;

use array::RustArray;
use fused::{
    layer_accumulate_gradient, layer_accumulate_gradient_batch,
    layer_adam_apply_accumulated_gradient, layer_apply_accumulated_gradient,
    layer_dropout_forward, layer_dropout_forward_batch, layer_dropout_hidden_delta,
    layer_dropout_hidden_delta_batch, layer_forward, layer_forward_batch, layer_hidden_delta,
    layer_hidden_delta_batch, layer_l2_apply_accumulated_gradient,
    layer_momentum_apply_accumulated_gradient, layer_output_delta, layer_relu_forward,
    layer_relu_forward_batch, layer_relu_hidden_delta, layer_relu_hidden_delta_batch,
    layer_softmax_forward, layer_softmax_forward_batch, layer_softmax_output_delta,
};
use linalg::outer;
use mnist::decode_mnist_pixels;
use random::{bernoulli_mask, uniform};
use ufuncs::{argmax, array_relu, array_relu_mask, array_softmax, exp, sum_axis0};

/// Proves the PyO3/maturin toolchain works end to end - importable and callable from Python,
/// nothing array-specific.
#[pyfunction]
fn ping() -> PyResult<String> {
    Ok("pong".to_string())
}

#[pymodule]
fn indrajala_ml_array(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(sum_axis0, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(array_relu, m)?)?;
    m.add_function(wrap_pyfunction!(array_relu_mask, m)?)?;
    m.add_function(wrap_pyfunction!(array_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(uniform, m)?)?;
    m.add_function(wrap_pyfunction!(bernoulli_mask, m)?)?;
    m.add_function(wrap_pyfunction!(decode_mnist_pixels, m)?)?;
    m.add_function(wrap_pyfunction!(layer_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layer_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_output_delta, m)?)?;
    m.add_function(wrap_pyfunction!(layer_hidden_delta, m)?)?;
    m.add_function(wrap_pyfunction!(layer_hidden_delta_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_accumulate_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(layer_accumulate_gradient_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_apply_accumulated_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(layer_adam_apply_accumulated_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(layer_l2_apply_accumulated_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(layer_momentum_apply_accumulated_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(layer_relu_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layer_relu_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_relu_hidden_delta, m)?)?;
    m.add_function(wrap_pyfunction!(layer_relu_hidden_delta_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_softmax_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layer_softmax_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_softmax_output_delta, m)?)?;
    m.add_function(wrap_pyfunction!(layer_dropout_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layer_dropout_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_dropout_hidden_delta, m)?)?;
    m.add_function(wrap_pyfunction!(layer_dropout_hidden_delta_batch, m)?)?;
    m.add_class::<RustArray>()?;
    Ok(())
}
