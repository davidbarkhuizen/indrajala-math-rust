use pyo3::prelude::*;

mod array;
mod attention;
mod batch_norm;
mod conv;
mod fused;
mod generator;
mod layer_norm;
mod linalg;
mod mnist;
mod ops;
mod random;
mod tokens;
mod ufuncs;

use array::RustArray;
use attention::{
    attention_accumulate_gradient_batch, attention_downstream_batch, attention_forward, attention_forward_batch,
};
use batch_norm::{
    batch_norm_accumulate_gradient_batch, batch_norm_downstream_batch, batch_norm_forward, batch_norm_forward_batch,
    linear_accumulate_gradient_batch, linear_forward, linear_forward_batch,
};
use conv::{
    conv_accumulate_gradient_batch, conv_downstream_batch, conv_forward_batch, conv_linear_accumulate_gradient_batch,
    conv_linear_forward_batch, max_pool_downstream_batch, max_pool_forward_batch, ConvGeometry,
};
use fused::{
    affine_forward, affine_forward_batch, layer_accumulate_gradient, layer_accumulate_gradient_batch,
    layer_adam_apply_accumulated_gradient, layer_apply_accumulated_gradient, layer_downstream, layer_downstream_batch,
    layer_dropout_forward, layer_dropout_forward_batch, layer_dropout_hidden_delta, layer_dropout_hidden_delta_batch,
    layer_dropout_hidden_delta_skip, layer_dropout_hidden_delta_skip_batch, layer_forward, layer_forward_batch,
    layer_hidden_delta, layer_hidden_delta_batch, layer_hidden_delta_skip, layer_hidden_delta_skip_batch,
    layer_l2_apply_accumulated_gradient, layer_momentum_apply_accumulated_gradient, layer_output_delta,
    layer_relu_forward, layer_relu_forward_batch, layer_relu_hidden_delta, layer_relu_hidden_delta_batch,
    layer_relu_hidden_delta_skip, layer_relu_hidden_delta_skip_batch, layer_sgd_step, layer_softmax_forward,
    layer_softmax_forward_batch, layer_softmax_output_delta,
};
use generator::{default_rng, Generator, SeedSequence};
use layer_norm::{
    layer_norm_accumulate_gradient_batch, layer_norm_downstream_batch, layer_norm_forward, layer_norm_forward_batch,
};
use linalg::{matmul_threads_for, outer, set_kernel_overrides, set_matmul_threading};
use mnist::decode_mnist_pixels;
use random::{bernoulli_mask, seed, seed_at_import, uniform};
use tokens::{
    embedding_accumulate_gradient, embedding_forward, patches_downstream, patches_forward, token_dropout_downstream,
    token_dropout_forward, token_mean_downstream, token_mean_forward,
};
use ufuncs::{
    argmax, array_dropout_mask, array_relu, array_relu_mask, array_sigmoid_mask, array_softmax, exp, sum_axis0,
};

/// Proves the PyO3/maturin toolchain works end to end - importable and callable from Python,
/// nothing array-specific.
#[pyfunction]
fn ping() -> PyResult<String> {
    Ok("pong".to_string())
}

// gil_used: pyo3 >= 0.28 declares a module free-threading safe unless told otherwise. This one
// hasn't been audited for free-threaded Python (Array's &mut self methods, the shared RNG
// state), so it keeps the GIL on 3.13t/3.14t.
#[pymodule(gil_used = true)]
fn indrajala_math_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(set_matmul_threading, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_threads_for, m)?)?;
    m.add_function(wrap_pyfunction!(set_kernel_overrides, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(sum_axis0, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(array_relu, m)?)?;
    m.add_function(wrap_pyfunction!(array_relu_mask, m)?)?;
    m.add_function(wrap_pyfunction!(array_sigmoid_mask, m)?)?;
    m.add_function(wrap_pyfunction!(array_dropout_mask, m)?)?;
    m.add_function(wrap_pyfunction!(array_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(seed, m)?)?;
    m.add_function(wrap_pyfunction!(random::random, m)?)?;
    m.add_function(wrap_pyfunction!(uniform, m)?)?;
    m.add_function(wrap_pyfunction!(bernoulli_mask, m)?)?;
    m.add_function(wrap_pyfunction!(default_rng, m)?)?;
    m.add_function(wrap_pyfunction!(decode_mnist_pixels, m)?)?;
    m.add_function(wrap_pyfunction!(layer_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layer_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_output_delta, m)?)?;
    m.add_function(wrap_pyfunction!(layer_hidden_delta, m)?)?;
    m.add_function(wrap_pyfunction!(layer_hidden_delta_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_accumulate_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(layer_accumulate_gradient_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_apply_accumulated_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(layer_sgd_step, m)?)?;
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
    m.add_function(wrap_pyfunction!(layer_downstream, m)?)?;
    m.add_function(wrap_pyfunction!(layer_downstream_batch, m)?)?;
    m.add_function(wrap_pyfunction!(affine_forward, m)?)?;
    m.add_function(wrap_pyfunction!(affine_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_hidden_delta_skip, m)?)?;
    m.add_function(wrap_pyfunction!(layer_hidden_delta_skip_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_relu_hidden_delta_skip, m)?)?;
    m.add_function(wrap_pyfunction!(layer_relu_hidden_delta_skip_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_dropout_hidden_delta_skip, m)?)?;
    m.add_function(wrap_pyfunction!(layer_dropout_hidden_delta_skip_batch, m)?)?;
    m.add_function(wrap_pyfunction!(conv_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(conv_downstream_batch, m)?)?;
    m.add_function(wrap_pyfunction!(conv_accumulate_gradient_batch, m)?)?;
    m.add_function(wrap_pyfunction!(conv_linear_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(conv_linear_accumulate_gradient_batch, m)?)?;
    m.add_function(wrap_pyfunction!(max_pool_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(max_pool_downstream_batch, m)?)?;
    m.add_function(wrap_pyfunction!(linear_forward, m)?)?;
    m.add_function(wrap_pyfunction!(linear_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(linear_accumulate_gradient_batch, m)?)?;
    m.add_function(wrap_pyfunction!(batch_norm_forward, m)?)?;
    m.add_function(wrap_pyfunction!(batch_norm_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(batch_norm_downstream_batch, m)?)?;
    m.add_function(wrap_pyfunction!(batch_norm_accumulate_gradient_batch, m)?)?;
    m.add_function(wrap_pyfunction!(patches_forward, m)?)?;
    m.add_function(wrap_pyfunction!(patches_downstream, m)?)?;
    m.add_function(wrap_pyfunction!(token_mean_forward, m)?)?;
    m.add_function(wrap_pyfunction!(token_mean_downstream, m)?)?;
    m.add_function(wrap_pyfunction!(embedding_forward, m)?)?;
    m.add_function(wrap_pyfunction!(embedding_accumulate_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(token_dropout_forward, m)?)?;
    m.add_function(wrap_pyfunction!(token_dropout_downstream, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm_downstream_batch, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm_accumulate_gradient_batch, m)?)?;
    m.add_function(wrap_pyfunction!(attention_forward, m)?)?;
    m.add_function(wrap_pyfunction!(attention_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(attention_downstream_batch, m)?)?;
    m.add_function(wrap_pyfunction!(attention_accumulate_gradient_batch, m)?)?;
    m.add_class::<RustArray>()?;
    m.add_class::<ConvGeometry>()?;
    m.add_class::<SeedSequence>()?;
    m.add_class::<Generator>()?;
    seed_at_import();
    Ok(())
}
