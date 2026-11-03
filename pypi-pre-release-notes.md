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

## Release process

- Releases to crates.io now use trusted publishing, with no stored token (#84). The API is
  unaffected.
