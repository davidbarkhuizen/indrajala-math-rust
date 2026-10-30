# indrajala-math-rust

A PyO3/maturin Rust extension providing a numpy-like `Array` type and a set of fused neural
network layer ops (forward/backward passes, Adam/L2/momentum/dropout variants). It is consumed by
[indrajala-ml](https://github.com/davidbarkhuizen/indrajala-ml) as the production backend for its
`*RustArray*` model classes, mounted there as a git submodule at `rust/`.

It is two crates in one Cargo workspace. `core/` is `indrajala-math`, a pure-Rust library with
the array type and every kernel (no pyo3, no dependencies); the root package (`src/`) is the Python
extension, thin bindings over it that read and build Python objects and lend the core their arrays
without a copy. `pyo3` is the bindings' only dependency; there are no Python runtime dependencies.

## Build and test

Requires [rustup](https://rustup.rs) and Python >= 3.14. `rust-toolchain.toml` pins the Rust
toolchain (with rustfmt and clippy), which rustup installs on first use; a distro `cargo` ignores
the pin. `maturin develop` installs into the
active virtualenv, so one must be active:

```
python3 -m venv .venv
source .venv/bin/activate
pip install maturin==1.15.0 pytest numpy
maturin develop --release   # rerun after any change under src/ or core/
pytest tests/
```

Always build with `--release`; a debug build is much slower.

Before committing, format and lint the Rust and the Python tests, and check the type stub (with
`ruff==0.16.9 pyright==1.1.414 mypy==2.3.1` installed, CI's pins):

```
cargo fmt
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p indrajala-math --release
ruff check --fix . && ruff format .
pyright
(cd /tmp && python -m mypy.stubtest indrajala_math_rust --allowlist "$OLDPWD/stubtest_allowlist.txt")
```

`rustfmt.toml` sets the Rust line width (120) and `Cargo.toml`'s `[lints]` the lint levels;
`pyproject.toml` configures ruff (120) and pyright (`standard`, over `tests/` and the stub).

`indrajala_math_rust.pyi` is the extension's type stub, which maturin packages with a `py.typed`
marker. Update it with any change to the Python API: stubtest (run on the installed build, after
`maturin develop`) fails on a missing, extra or renamed function or parameter, and pyright on the
tests catches wrong types.

CI (`.github/workflows/ci.yml`) runs all of these, and the build and test steps, on every push and
PR to `main`.

`tests/` (~3,000 tests, about 40 s) checks each op against numpy through the bindings and needs
nothing from indrajala-ml. `core/tests/` is the core's own: its API contract, numpy's published
generator draws, and each kernel family's output bits against `core/tests/fixtures/bits.txt`, which
every platform must reproduce (`INDRAJALA_MATH_BLESS=1 cargo test -p indrajala-math` rewrites it
after a change meant to move bits). The tests that check the fused layer ops against
indrajala-ml's numpy reference classes live in indrajala-ml's own `tests/`, so a change to
`core/src/fused.rs` or `core/src/conv.rs` should also be tested there: in an indrajala-ml checkout, point `rust/` at the new commit, then
`./cli build-rust && ./cli test`.

## Layout

Each `src/<module>.rs` binds `core/src/<module>.rs`, which holds the module's arithmetic (its
`#[pyfunction]`s are wrappers over the core's functions of the same names), except `src/lib.rs`
(the Python module) and the core's `error.rs` (its `Error`, raised as `ValueError` or
`OverflowError`). The Python API, by module:

| File | Python API |
| --- | --- |
| `src/lib.rs` | module definition; `ping()` toolchain check |
| `indrajala_math_rust.pyi` | type stub for the whole Python API (packaged with `py.typed`) |
| `src/array.rs` | `RustArray`: construction, `zeros`, `shape`, `.T`, indexing and contiguous slicing, `reshape`, `copy`, `tolist` |
| `src/ops.rs` | elementwise `+ - * /` and in-place `+= -=` |
| `src/linalg.rs` | `@` (matmul), `outer`; `set_matmul_threading(max_threads, threshold_flops)`, a test/benchmark override of the matmul threading (0 = default); `matmul_threads_for(m, k, n)`, the thread count the policy picks for an `(m, k) @ (k, n)` product |
| `src/ufuncs.rs` | `exp`, `sum_axis0`, `argmax`, `array_relu`, `array_relu_mask`, `array_sigmoid_mask`, `array_dropout_mask` (a hidden delta from a downstream the layer is handed, before a layer with no `W`), `array_softmax` |
| `src/random.rs` | `seed`, `random`, `uniform`, `bernoulli_mask`: numpy's legacy `np.random` (MT19937) in a separate state; after `seed(s)`, bit-identical to numpy's draws after `np.random.seed(s)` |
| `src/generator.rs` | `SeedSequence`, `Generator`, `default_rng`: numpy's `default_rng` (`Generator(PCG64(SeedSequence(seed)))`), each generator its own state; bit-identical to numpy's draws from the same seed, with numpy's `bit_generator.state` as its `state` |
| `src/mnist.rs` | `decode_mnist_pixels`: raw MNIST records to normalised pixels |
| `src/fused.rs` | `layer_*`: one call per `ArrayLayer` method (forward, deltas, downstream, gradient accumulate/apply), single-example and `_batch`; the dropout forwards draw their mask from an optional `rng: Generator`, else from `random.rs`'s global; `affine_forward*` and the `layer_*hidden_delta_skip*` ops of residual blocks (`(body_delta @ body_W + skip) * f'(a)`, the layer before a block's fork) |
| `src/conv.rs` | `ConvGeometry`; `conv_*_batch`/`max_pool_*_batch`: one call per `ConvArrayLayer`/`MaxPoolArrayLayer` method (forward, downstream, gradient accumulate), batch-only; `conv_linear_*_batch`, `LinearConvArrayLayer`'s bias-free forward and accumulate |
| `src/batch_norm.rs` | `linear_*`, `batch_norm_*`: one call per `LinearArrayLayer`/`BatchNormArrayLayer` method (batch normalization, dense or per conv channel with `positions`: forward in training and inference, downstream, gradient accumulate) |
| `src/tokens.rs` | `patches_*`, `token_mean_*`, `embedding_*`, `token_dropout_*`: one call per `PatchesArrayLayer`/`TokenMeanArrayLayer`/`EmbeddingArrayLayer`/`TokenDropoutArrayLayer` method (forward, downstream), one example or a batch; `Position` is `+` and `sum_axis0` |
| `src/layer_norm.rs` | `layer_norm_*`: one call per `LayerNormArrayLayer` method (forward, single and batch; downstream, gradient accumulate) |
| `src/attention.rs` | `attention_*`: one call per pass of `AttentionArrayLayer` (forward, single and batch; downstream, which also returns `dQ, dK, dV`; gradient accumulate), multi-head (`heads`), causal (`causal`), dropping its weights in training (`dropout`, `training`, `rng`; the downstream takes the `mask`), rotary positions (`cos`, `sin`: the forward ops rotate `Q` and `K`, the downstream rotates `dQ` and `dK` back), each a thin wrapper over the crate-internal blocks project, attend and combine, composed of the crate's dense ops, `@` and `array_softmax` |
| `src/recurrent.rs` | `state_input_*`, `attention_pooling_*`, `gru_*`, `codes_*`: one call per pass of a window-recurrent network's `StateInputArrayLayer`/`AttentionPoolingArrayLayer`/`GRUArrayLayer`/`CodesArrayLayer` (forward, backward), batch-only, composed of the crate's dense ops, `@` and `array_softmax`, with `tanh` through `exp`; `concatenate_columns`, which joins the windows' outputs |
| `core/src/cr_exp.rs` | (core only) `exp`, correctly rounded: a port of CORE-MATH's `cr_exp`; every `exp` in the crate (the `exp` ufunc, sigmoid, `tanh`, softmax) goes through it |
| `core/src/scalar.rs` | (core only) the scalar rules below: relu, the softmax row maximum, `powi` |

Each `layer_*` function in `core/src/fused.rs` mirrors a method of indrajala-ml's
`indrajala_ml/model/layers/numpy/array_layer.py` or one of its ReLU/softmax/dropout/Adam/L2/momentum variants,
and must stay numerically identical to it. Likewise each `conv_*`/`max_pool_*` function in
`core/src/conv.rs` mirrors a method of `indrajala_ml/model/layers/numpy/conv_array_layer.py`/`max_pool_array_layer.py`, each `conv_linear_*` function one of its `LinearConvArrayLayer`, each `linear_*`/`batch_norm_*` function in `core/src/batch_norm.rs` one of `linear_array_layer.py`/`batch_norm_array_layer.py`, and each function in `core/src/tokens.rs`, `core/src/layer_norm.rs`, `core/src/attention.rs` and `core/src/recurrent.rs` one of `token_array_layer.py`, `layer_norm_array_layer.py`, `attention_array_layer.py` or `recurrent_array_layer.py`. Conv tensors cross the boundary as matrices
(`Array` stays 1D/2D); `core/src/conv.rs`'s module comment gives the layouts.

## Same bits on every machine

A network on this backend trains to the same bits on x86_64 and aarch64 (indrajala-ml's
retired aarch64 workplan: `git show 6b9c6fe:docs/workplans/infrastructure/aarch64-workplan.md`). Products follow one order on every path: four `mul_add` lanes
combined `(l0 + l1) + (l2 + l3)`, tails in order, reductions as left folds (`core/src/linalg.rs`);
threading splits rows or column chunks, never one sum. `exp` is correctly rounded
(`core/src/cr_exp.rs`), so it is mathematics' value, not a C library's. A C library's `exp` need not
be: glibc 2.43's rounds about 0.075% of inputs to the neighbouring double (worst 0.504 ULP,
within its documented bound), and which inputs differs between libraries. Where IEEE 754 or the compiler
would leave a result to the machine, `core/src/scalar.rs` fixes it: relu gives `+0.0` for both zeros
and propagates NaN; the softmax row maximum is a left fold with relu's comparison; Adam's `powi`
is square-and-multiply from the exponent's low bit. Not claimed: the sign and payload of a NaN
that an invalid operation (`inf - inf`, `0 * inf`) creates, which is the hardware's.

## Releasing

One version names both packages: `[workspace.package] version` in `Cargo.toml` (the PyPI package
reads it through maturin). `.github/workflows/release.yml` publishes on a tag, after the same
build-and-test matrix every PR runs; the `crates-io` and `pypi` environments wait for the owner's
approval.

1. Bump the version in `Cargo.toml` by PR, and merge it.
2. Rehearse on TestPyPI: tag `main` `testpypi-v<version>` and push the tag. Check the run, the
   TestPyPI page (the README, the metadata, a file per platform) and the verify jobs. Neither index
   takes a version twice, even after a deletion: a failed rehearsal is fixed by PR and a new version.
3. Tag `main` `v<version>` and push the tag; approve `crates-io`, then `pypi`. The run publishes the
   core to crates.io, then the wheels and sdist to PyPI, installs both from the indexes and tests
   them, and creates the GitHub Release with the wheels, the sdist and the `.crate`.

A run that fails after an index has taken the version is re-run (`gh run rerun --failed`), never
retagged: each publish skips a version its index already has.

## License

MIT, see [LICENSE](LICENSE). `core/src/cr_exp.rs` is a port of CORE-MATH's `cr_exp` (MIT, Alexei
Sibidanov), its copyright and permission notice kept in the file.
