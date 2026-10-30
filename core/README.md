# indrajala-math

A 1D/2D `f64` array and the kernels of neural network layers, in pure Rust with no dependencies,
written as the backend of [indrajala-ml](https://github.com/davidbarkhuizen/indrajala-ml). Its
Python bindings are on PyPI as
[indrajala-math-rust](https://pypi.org/project/indrajala-math-rust/).

```toml
[dependencies]
indrajala-math = "0.1"
```

```rust
use indrajala_math::linalg::matmul;
use indrajala_math::Array;

let w = Array::from_matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
let x = Array::from_vector(vec![1.0, 0.0, -1.0]);
let y = matmul(&w, &x).unwrap();
assert_eq!(y.data(), &[-2.0, -2.0]);
```

## What it has

- **Products** (`linalg`): `a @ b` in its three shape cases, `c + a @ b`, `a @ b.T`, a long-`K`
  product, `outer`, with AVX2/FMA kernels on x86_64 (picked at runtime, scalar fallback) and NEON
  on aarch64, threaded over output rows for large products.
- **Layers**, forward and backward: dense and fused layer updates (SGD, momentum, L2, Adam,
  dropout), conv and max pooling, batch and layer norm, multi-head attention (causal, dropout,
  rotary positions), tokens (patches, embeddings, means, dropout), a GRU cell, attention pooling,
  and vector-quantized codes, plain and factored.
- **Functions**: a correctly rounded `exp` (a port of
  [CORE-MATH](https://core-math.gitlabpages.inria.fr/)'s `cr_exp`), softmax, ReLU and their masks.
- **Generators**: numpy's legacy `np.random` (MT19937) and `default_rng`
  (`Generator(PCG64(SeedSequence(seed)))`), bit for bit: after the same seed, the same draws as
  numpy.

## Same bits on every machine

Every reduction has one fixed summation order, so an operation's result has the same bits on
x86_64 and aarch64, with or without its SIMD kernels and at any thread count. The bits are within
rounding of numpy, not identical to it. The tests record each kernel family's output bits, and CI
checks them on every platform.

## Stability

The API follows indrajala-ml's needs, and there is no stability promise before 1.0. Shape and
argument errors come back as [`Error`](https://docs.rs/indrajala-math/latest/indrajala_math/enum.Error.html);
the only panics are the ones documented (`Array::from_matrix` on a buffer of the wrong length).

## License

MIT. `src/cr_exp.rs` is a port of CORE-MATH's `cr_exp` (MIT, Alexei Sibidanov), its copyright and
permission notice kept in the file.
