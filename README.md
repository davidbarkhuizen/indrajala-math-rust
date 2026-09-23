# indrajala-math-rust

A PyO3/maturin Rust extension providing a numpy-like `Array` type and a set of fused neural
network layer ops (forward/backward passes, Adam/L2/momentum/dropout variants). It is consumed by
[indrajala-ml](https://github.com/davidbarkhuizen/indrajala-ml) as the production backend for its
`*RustArray*` model classes, mounted there as a git submodule at `rust/`.

`pyo3` is the only Rust dependency; there are no Python runtime dependencies.

## Build and test

Requires a Rust toolchain (`cargo`) and Python >= 3.9. `maturin develop` installs into the
active virtualenv, so one must be active:

```
python3 -m venv .venv
source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release   # rerun after any change under src/
pytest tests/
```

Always build with `--release`; a debug build is much slower. CI (`.github/workflows/ci.yml`) runs
the same steps, plus `cargo check --release`, on every push and PR to `main`.

`tests/` (~200 tests, under a second) checks each op against numpy and needs nothing from
indrajala-ml. The tests that check the fused layer ops against indrajala-ml's numpy reference
classes live in indrajala-ml's own `tests/`, so a change to `src/fused.rs` should also be tested
there: in an indrajala-ml checkout, point `rust/` at the new commit, then
`./cli build-rust && ./cli test`.

## Layout

| File | Python API |
| --- | --- |
| `src/lib.rs` | module definition; `ping()` toolchain check |
| `src/array.rs` | `RustArray`: construction, `zeros`, `shape`, `.T`, indexing and contiguous slicing, `reshape`, `copy`, `tolist` |
| `src/ops.rs` | elementwise `+ - * /` and in-place `+= -=` |
| `src/linalg.rs` | `@` (matmul), `outer` |
| `src/ufuncs.rs` | `exp`, `sum_axis0`, `argmax`, `array_relu`, `array_relu_mask`, `array_softmax` |
| `src/random.rs` | `uniform`, `bernoulli_mask`: unseeded hand-rolled xorshift128+, so not reproducible against numpy |
| `src/mnist.rs` | `decode_mnist_pixels`: raw MNIST records to normalised pixels |
| `src/fused.rs` | `layer_*`: one call per `ArrayLayer` method (forward, deltas, gradient accumulate/apply), single-example and `_batch` |

Each `layer_*` function in `src/fused.rs` mirrors a method of indrajala-ml's
`indrajala_ml/model/array_layer.py` or one of its ReLU/softmax/dropout/Adam/L2/momentum variants,
and must stay numerically identical to it.
