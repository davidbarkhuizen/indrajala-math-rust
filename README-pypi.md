# indrajala-math-rust

A numpy-like `Array` and fused neural network layer ops, written in Rust as the backend of
[indrajala-ml](https://github.com/davidbarkhuizen/indrajala-ml). Prebuilt wheels: no Rust
toolchain is needed.

```
pip install indrajala-math-rust
```

```python
import indrajala_math_rust as im

w = im.Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
x = im.Array([1.0, 0.0, -1.0])
print((w @ x).tolist())  # [-2.0, -2.0]
rng = im.default_rng(42)
print(rng.random(3).tolist())  # numpy's default_rng(42).random(3), bit for bit
```

## What it has

- `Array`: 1D and 2D `float64`, `@`, elementwise arithmetic, `.T`, indexing and contiguous slices,
  `reshape`, `tolist`.
- One call per layer pass, forward and backward: dense layers and their updates (SGD, momentum, L2,
  Adam, dropout), conv and max pooling, batch and layer norm, multi-head attention (causal,
  dropout, rotary positions), tokens, a GRU cell, attention pooling and vector-quantized codes.
- A correctly rounded `exp`, softmax, ReLU and their masks.
- numpy's generators bit for bit: `seed`/`random`/`uniform` as legacy `np.random` (MT19937), and
  `default_rng`, `Generator` and `SeedSequence` as numpy's PCG64.

The type stub (`py.typed`) covers the whole API.

## Same bits on every machine

Every reduction has one fixed summation order, so results have the same bits on x86_64 and
aarch64, with or without the SIMD kernels and at any thread count: a network trains to the same
bits everywhere. They are within rounding of numpy, not identical to it.

## Platforms

CPython 3.14 or later, with wheels for Linux (glibc and musl, x86_64 and aarch64), macOS (arm64
and x86_64) and Windows (x86_64 and arm64), each built and tested on its own platform.

## Stability

The API follows indrajala-ml's needs, and there is no stability promise before 1.0.

The kernels are also a pure-Rust crate,
[indrajala-math](https://crates.io/crates/indrajala-math). Source, issues and the contributor's
guide: [github.com/davidbarkhuizen/indrajala-math-rust](https://github.com/davidbarkhuizen/indrajala-math-rust).
MIT licensed.
