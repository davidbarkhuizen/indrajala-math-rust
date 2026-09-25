# indrajala-math-rust

A PyO3/maturin Rust extension providing a numpy-like `Array` type and a set of fused neural
network layer ops (forward/backward passes, Adam/L2/momentum/dropout variants). It is consumed by
[indrajala-ml](https://github.com/davidbarkhuizen/indrajala-ml) as the production backend for its
`*RustArray*` model classes, mounted there as a git submodule at `rust/`.

`pyo3` is the only Rust dependency; there are no Python runtime dependencies.

## Build and test

Requires [rustup](https://rustup.rs) and Python >= 3.9. `rust-toolchain.toml` pins the Rust
toolchain (with rustfmt and clippy), which rustup installs on first use; a distro `cargo` ignores
the pin. `maturin develop` installs into the
active virtualenv, so one must be active:

```
python3 -m venv .venv
source .venv/bin/activate
pip install maturin==1.15.0 pytest numpy
maturin develop --release   # rerun after any change under src/
pytest tests/
```

Always build with `--release`; a debug build is much slower.

Before committing, format and lint the Rust and the Python tests, and check the type stub (with
`ruff==0.16.9 pyright==1.1.414 mypy==2.3.1` installed, CI's pins):

```
cargo fmt
cargo clippy --all-targets -- -D warnings
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

`tests/` (~1,560 tests, about 20 s) checks each op against numpy and needs nothing from
indrajala-ml. The tests that check the fused layer ops against indrajala-ml's numpy reference
classes live in indrajala-ml's own `tests/`, so a change to `src/fused.rs` or `src/conv.rs` should
also be tested there: in an indrajala-ml checkout, point `rust/` at the new commit, then
`./cli build-rust && ./cli test`.

## Layout

| File | Python API |
| --- | --- |
| `src/lib.rs` | module definition; `ping()` toolchain check |
| `indrajala_math_rust.pyi` | type stub for the whole Python API (packaged with `py.typed`) |
| `src/array.rs` | `RustArray`: construction, `zeros`, `shape`, `.T`, indexing and contiguous slicing, `reshape`, `copy`, `tolist` |
| `src/ops.rs` | elementwise `+ - * /` and in-place `+= -=` |
| `src/linalg.rs` | `@` (matmul), `outer`; `set_matmul_threading(max_threads, threshold_flops)`, a test/benchmark override of the matmul threading (0 = default); `matmul_threads_for(m, k, n)`, the thread count the policy picks for an `(m, k) @ (k, n)` product |
| `src/ufuncs.rs` | `exp`, `sum_axis0`, `argmax`, `array_relu`, `array_relu_mask`, `array_softmax` |
| `src/random.rs` | `uniform`, `bernoulli_mask`: unseeded hand-rolled xorshift128+, so not reproducible against numpy |
| `src/mnist.rs` | `decode_mnist_pixels`: raw MNIST records to normalised pixels |
| `src/fused.rs` | `layer_*`: one call per `ArrayLayer` method (forward, deltas, downstream, gradient accumulate/apply), single-example and `_batch` |
| `src/conv.rs` | `ConvGeometry`; `conv_*_batch`/`max_pool_*_batch`: one call per `ConvArrayLayer`/`MaxPoolArrayLayer` method (forward, downstream, gradient accumulate), batch-only |

Each `layer_*` function in `src/fused.rs` mirrors a method of indrajala-ml's
`indrajala_ml/model/array_layer.py` or one of its ReLU/softmax/dropout/Adam/L2/momentum variants,
and must stay numerically identical to it. Likewise each `conv_*`/`max_pool_*` function in
`src/conv.rs` mirrors a method of `indrajala_ml/model/conv_array_layer.py`/`max_pool_array_layer.py`. Conv tensors cross the boundary as matrices
(`Array` stays 1D/2D); `src/conv.rs`'s module comment gives the layouts.
