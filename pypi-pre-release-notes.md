# PyPI pre-release notes

Changes since the last PyPI release (0.1.0) to note in the next release's notes. Add entries as
they merge, then move them into the release notes at release time and empty this file.

## Breaking changes

- **ReLU, GELU and GELU-tanh are one fused family** (#90; indrajala-ml's FFN activations
  workplan, D9). This removes the old ReLU names and passes the activation by name (`"relu"`,
  `"gelu"`, `"gelu_tanh"`):

  | Removed | Replacement |
  | --- | --- |
  | `layer_relu_forward(w, x, b)` | `layer_activation_forward(w, x, b, "relu")` |
  | `layer_relu_forward_batch(w, x, b)` | `layer_activation_forward_batch(w, x, b, "relu")` |
  | `layer_relu_hidden_delta(next_w, next_delta, a)` | `layer_activation_hidden_delta(next_w, next_delta, a, "relu")` |
  | `layer_relu_hidden_delta_batch(...)` | `layer_activation_hidden_delta_batch(..., "relu")` |
  | `layer_relu_hidden_delta_skip(...)` | `layer_activation_hidden_delta_skip(..., "relu")` |
  | `layer_relu_hidden_delta_skip_batch(...)` | `layer_activation_hidden_delta_skip_batch(..., "relu")` |
  | `array_relu_mask(downstream, a)` | `array_activation_mask(downstream, a, "relu")` |

  - **The forwards return a tuple:** `layer_activation_forward*` returns `(a, z)` rather than
    `a`. For ReLU, `z` is `None`; for the GELUs it is the pre-activation, which their derivative
    reads.
  - **What the deltas read:** the hidden deltas and the mask take that `cached` operand, `a` for
    ReLU and `z` for the GELUs.
  - **Unknown names:** an unknown activation name raises `ValueError`.
  - **ReLU's results are unchanged:** its arithmetic is as before, so its bits are unchanged.
  - **Speed:** the single-example ReLU hidden delta is about 0.2 µs a call slower (indrajala-ml
    #1183's A/B). Batched ops are unchanged.

- **Attention is tiled in key blocks** (#88; indrajala-ml's context-length workplan, D3, D12): an
  online softmax that keeps each row's maximum `m` and sum `l` instead of the weights `P`.
  - `attention_forward(...)` and `attention_forward_batch(...)` returned
    `(a, q, k, v, p, h, mask)`. They now return `(a, q, k, v, h, m, l, dropout_rng)`: `m` and `l`
    are `(T, heads)`, and `dropout_rng` is a copy of `rng` before the draws (or `None`). Both gain
    keyword `segments=None` and `block=16`.
  - `attention_downstream_batch(...)` takes `h, m, l` in place of `p`, and `causal=False` and
    `segments=None` in place of `mask=None`. With dropout it takes `dropout_rng` and draws the same
    masks again. It also gains `block=16`.

## Additions

- **`attention_step`** (#85): one token's causal attention on a key/value cache, for generation.
  It returns `(a, k_cache, v_cache)`.
- **A segment mask** (#87): `segments=` on `attention_forward*` and `attention_downstream_batch`,
  so that tokens attend only within their segment.
- **AdamW** (#86): `layer_adam_apply_accumulated_gradient(..., weight_decay=0.0)`, decoupled
  weight decay on `W`. The default leaves Adam as it was.
- **GELU and GELU-tanh** (`"gelu"`, `"gelu_tanh"`) in the fused family above (#90). GELU is
  exact, through the crate's `erf`. GELU-tanh is GPT-2's approximation. Both go through the
  correctly rounded `exp`, so their bits are the same on every machine.
- **SiLU and the gated FFN** (`"silu"`; `gated_forward_batch`, `gated_delta_batch`): `act(a) * v`
  from one `(2 * size, d)` product, the gate's rows then the value's (SwiGLU with `"silu"`, GEGLU
  with a GELU), and its pre-activation's delta, for indrajala-ml's `GatedDense`.
- **Grouped-query attention** (indrajala-ml's grouped-query attention workplan, D4): keyword
  `key_value_heads=None` on `attention_forward*`, `attention_step` and
  `attention_downstream_batch`. With `g` key/value heads dividing `heads`, `wk`, `bk`, `wv`, `bv`
  and the caches `k`, `v` (and `dk`, `dv`, `attention_step`'s `k_cache`, `v_cache`) are `g * d_k`
  wide, query head `i` reading key/value head `i // (heads / g)`; `g = 1` is multi-query
  attention. `None` means `heads`: every existing call is unchanged, with the same bits.
- **Cross-attention** (indrajala-ml's cross-attention workplan, D3, D11, D13):
  `cross_attention_forward_batch`, `cross_attention_downstream_batch` and
  `cross_attention_accumulate_gradient_batch`. The queries come from `x`, `(N, T * d)`, and the keys
  and values from `memory`, `(N, S * d_m)`, so `wk` and `wv` are `(g * d_k, d_m)`. The ops are never
  causal and never rotary. With `segments` and `memory_segments` (both or neither), a query reads
  only its own segment's keys, with no own-column exception, so a pad's query reads no key: its `h`
  is 0 and its output is `bo`. The downstream returns `(dx, dm, dq, dk, dv)`, where
  `dm = dK Wk + dV Wv` is the memory's gradient as `(N * S, d_m)` rows. Self-attention's bits are
  unchanged.
- **Cross-attention's decoding** (indrajala-ml's cross-attention workplan, D7):
  `cross_attention_attend_batch(x, k, v, wq, bq, wo, bo, heads, ...)` attends queries against keys
  and values computed once per source (the memory's rows through `wk`, `bk` and `wv`, `bv`). Each
  query's bits are the forward pass's at that token.
- **`erf`** (#89), a port of fdlibm's `s_erf.c` through the correctly rounded `exp`. It is within
  one ulp, but not correctly rounded.

## crates.io (`indrajala-math`): breaking changes

The next release is 0.2.0 (indrajala-ml's crate structure workplan, D4). These changes are to the
Rust crate only. The Python module's names and signatures are unchanged (D3), and so is every
result's bits (D5).

- **The modules are a layered tree** (D1): `num` (scalar rules), then `ops`, `linalg` and `random`,
  then `nn` (the layers), then `program` and `data`. Each module uses only its own layer and those
  below it, and `core/tests/layers.rs` enforces that. There are no aliases at the old paths.

  | 0.1.0 module | 0.2.0 |
  | --- | --- |
  | `activation` | `num::activation` |
  | `cr_exp` | `num::exp` (with `recurrent`'s `tanh_in_place`, crate-private) |
  | `erf` | `num::erf` |
  | `scalar` | `num::scalar` |
  | `ops` | `ops::arith` |
  | `ufuncs` | `ops::unary`; its masks `ops::mask` |
  | `random` | `random::legacy` |
  | `generator` | `random::generator` |
  | `fused` | `nn::dense`, `nn::dropout`, `nn::softmax`, `nn::gated`, `nn::optim` |
  | `batch_norm`, `layer_norm` | `nn::norm::batch_norm`, `nn::norm::layer_norm` |
  | `conv`, `tokens`, `recurrent` | `nn::conv`, `nn::tokens`, `nn::recurrent` |
  | `attention` | `nn::attention`; cross-attention `nn::attention::cross` |
  | `factored` | `nn::recurrent::factored` |
  | `mnist` | `data::mnist` |
  | `linalg`, `program`, `Array`, `Shape`, `Error` | unchanged |

- **Names drop their module's prefix** (D2). The functions with new names:

  | 0.1.0 | 0.2.0 |
  | --- | --- |
  | `fused::layer_forward` | `nn::dense::forward` |
  | `fused::layer_forward_batch` | `nn::dense::forward_batch` |
  | `fused::layer_output_delta` | `nn::dense::output_delta` |
  | `fused::layer_downstream` | `nn::dense::downstream` |
  | `fused::layer_downstream_batch` | `nn::dense::downstream_batch` |
  | `fused::layer_hidden_delta` | `nn::dense::hidden_delta` |
  | `fused::layer_hidden_delta_batch` | `nn::dense::hidden_delta_batch` |
  | `fused::layer_accumulate_gradient` | `nn::dense::accumulate_gradient` |
  | `fused::layer_accumulate_gradient_batch` | `nn::dense::accumulate_gradient_batch` |
  | `fused::layer_activation_forward` | `nn::dense::activation_forward` |
  | `fused::layer_activation_forward_batch` | `nn::dense::activation_forward_batch` |
  | `fused::layer_activation_hidden_delta` | `nn::dense::activation_hidden_delta` |
  | `fused::layer_activation_hidden_delta_batch` | `nn::dense::activation_hidden_delta_batch` |
  | `fused::layer_hidden_delta_skip` | `nn::dense::hidden_delta_skip` |
  | `fused::layer_hidden_delta_skip_batch` | `nn::dense::hidden_delta_skip_batch` |
  | `fused::layer_activation_hidden_delta_skip` | `nn::dense::activation_hidden_delta_skip` |
  | `fused::layer_activation_hidden_delta_skip_batch` | `nn::dense::activation_hidden_delta_skip_batch` |
  | `fused::layer_apply_accumulated_gradient` | `nn::optim::apply_accumulated_gradient` |
  | `fused::layer_sgd_step` | `nn::optim::sgd_step` |
  | `fused::layer_adam_apply_accumulated_gradient` | `nn::optim::adam_apply_accumulated_gradient` |
  | `fused::layer_l2_apply_accumulated_gradient` | `nn::optim::l2_apply_accumulated_gradient` |
  | `fused::layer_momentum_apply_accumulated_gradient` | `nn::optim::momentum_apply_accumulated_gradient` |
  | `fused::layer_softmax_forward` | `nn::softmax::forward` |
  | `fused::layer_softmax_forward_batch` | `nn::softmax::forward_batch` |
  | `fused::layer_softmax_output_delta` | `nn::softmax::output_delta` |
  | `fused::layer_dropout_forward` | `nn::dropout::forward` |
  | `fused::layer_dropout_forward_batch` | `nn::dropout::forward_batch` |
  | `fused::layer_dropout_hidden_delta` | `nn::dropout::hidden_delta` |
  | `fused::layer_dropout_hidden_delta_batch` | `nn::dropout::hidden_delta_batch` |
  | `fused::layer_dropout_hidden_delta_skip` | `nn::dropout::hidden_delta_skip` |
  | `fused::layer_dropout_hidden_delta_skip_batch` | `nn::dropout::hidden_delta_skip_batch` |
  | `fused::gated_forward_batch` | `nn::gated::forward_batch` |
  | `fused::gated_delta_batch` | `nn::gated::delta_batch` |
  | `batch_norm::batch_norm_forward` | `nn::norm::batch_norm::forward` |
  | `batch_norm::batch_norm_forward_batch` | `nn::norm::batch_norm::forward_batch` |
  | `batch_norm::batch_norm_downstream_batch` | `nn::norm::batch_norm::downstream_batch` |
  | `batch_norm::batch_norm_accumulate_gradient_batch` | `nn::norm::batch_norm::accumulate_gradient_batch` |
  | `layer_norm::layer_norm_forward` | `nn::norm::layer_norm::forward` |
  | `layer_norm::layer_norm_forward_batch` | `nn::norm::layer_norm::forward_batch` |
  | `layer_norm::layer_norm_downstream_batch` | `nn::norm::layer_norm::downstream_batch` |
  | `layer_norm::layer_norm_accumulate_gradient_batch` | `nn::norm::layer_norm::accumulate_gradient_batch` |
  | `conv::conv_forward_batch` | `nn::conv::forward_batch` |
  | `conv::conv_downstream_batch` | `nn::conv::downstream_batch` |
  | `conv::conv_accumulate_gradient_batch` | `nn::conv::accumulate_gradient_batch` |
  | `conv::conv_linear_forward_batch` | `nn::conv::linear_forward_batch` |
  | `conv::conv_linear_accumulate_gradient_batch` | `nn::conv::linear_accumulate_gradient_batch` |
  | `conv::ConvGeometry` | `nn::conv::Geometry` |
  | `attention::attention_forward` | `nn::attention::forward` |
  | `attention::attention_forward_batch` | `nn::attention::forward_batch` |
  | `attention::attention_step` | `nn::attention::step` |
  | `attention::attention_downstream_batch` | `nn::attention::downstream_batch` |
  | `attention::attention_accumulate_gradient_batch` | `nn::attention::accumulate_gradient_batch` |
  | `attention::cross_attention_forward_batch` | `nn::attention::cross::forward_batch` |
  | `attention::cross_attention_attend_batch` | `nn::attention::cross::attend_batch` |
  | `attention::cross_attention_downstream_batch` | `nn::attention::cross::downstream_batch` |
  | `attention::cross_attention_accumulate_gradient_batch` | `nn::attention::cross::accumulate_gradient_batch` |
  | `tokens::token_mean_forward` | `nn::tokens::mean_forward` |
  | `tokens::token_mean_downstream` | `nn::tokens::mean_downstream` |
  | `tokens::token_dropout_forward` | `nn::tokens::dropout_forward` |
  | `tokens::token_dropout_downstream` | `nn::tokens::dropout_downstream` |
  | `factored::factored_codes_forward` | `nn::recurrent::factored::codes_forward` |
  | `factored::factored_codes_backward` | `nn::recurrent::factored::codes_backward` |
  | `ufuncs::array_relu` | `ops::unary::relu` |
  | `ufuncs::array_softmax` | `ops::unary::softmax` |
  | `ufuncs::array_activation_mask` | `ops::mask::activation` |
  | `ufuncs::array_sigmoid_mask` | `ops::mask::sigmoid` |
  | `ufuncs::array_dropout_mask` | `ops::mask::dropout` |
  | `mnist::decode_mnist_pixels` | `data::mnist::decode_pixels` |
  | `program::window_program_forward` | `program::forward` |
  | `program::window_program_learn` | `program::learn` |
  | `batch_norm::linear_forward`, `linear_forward_batch`, `linear_accumulate_gradient_batch` | `nn::dense::` (same names) |

- **`num::activation::Activation` gains `Sigmoid`**, batch norm's activation, so the crate has
  one activation enum (it replaces batch norm's private one). A `match` on `Activation` needs the
  new arm. `from_name` doesn't name it: the hidden layers' activation ops take the FFN family only.
  It also gains `apply_in_place`, with the same bits as `apply`.

## Release process

- Releases to crates.io now use trusted publishing, with no stored token (#84). The API is
  unaffected.
