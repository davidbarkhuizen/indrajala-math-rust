# indrajala-math-rust

A PyO3/maturin Rust extension providing a numpy-like `Array` type and a set of fused neural
network layer ops (forward/backward passes, Adam/L2/momentum/dropout variants). It is consumed by
[indrajala-ml](https://github.com/davidbarkhuizen/indrajala-ml) as the production backend for its
`*RustArray*` model classes, mounted there as a git submodule at `rust/`.

## Build

```
maturin develop --release
```

## Test

```
pip install pytest numpy
pytest tests/
```

This test suite is self-contained (numpy only) and does not depend on indrajala-ml. Integration
tests that parity-check this crate's ops against indrajala-ml's own pure-Python reference model
classes live in indrajala-ml's own `tests/` directory instead.
