# indrajala-math-rust

The Rust backend of [indrajala-ml](https://github.com/davidbarkhuizen/indrajala-ml), which vendors
this crate as its `rust/` submodule. That repository's `CLAUDE.md` and `docs/measurement.md` set
the workflow and the timing rules; they apply here too.

- **One focused PR per chunk of work, from a feature branch**, squash-merged once CI passes.
- **A crate change lands here first**, with this repo's own numpy-only tests; then a "Bump rust/"
  PR in indrajala-ml runs its full suite (`./cli build-rust && ./cli test`). A change to
  `src/fused.rs` or `src/conv.rs` must pass there too, before either merges.
- **Bit-identical claims are tested:** a test pins the new op with `==` on `tolist()` at shapes
  that reach every kernel path, threading and blocking threshold (indrajala-ml's
  `docs/measurement.md`, §9).
- **The toolchain is pinned** by `rust-toolchain.toml` (1.98.1). `rustup toolchain install` of
  another version makes that one the default; reset with `rustup default 1.98.1`.
- Build and lint as the README's "Build and test" says; always `maturin develop --release`.
