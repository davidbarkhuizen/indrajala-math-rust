#![warn(missing_docs)]
//! The array type and kernels behind [indrajala-math-rust](https://pypi.org/project/indrajala-math-rust/),
//! indrajala-ml's backend, as a pure-Rust library. Every operation borrows its operands
//! (`&Array`) and returns a new [`Array`]; bindings that hold an `Array` inside their own type copy
//! nothing at the boundary.
//!
//! Every reduction has one fixed summation order, so an operation gives the same bits on every
//! machine and with or without its SIMD kernels (AVX2/FMA on x86_64, NEON on aarch64), and within
//! rounding of numpy rather than bit-identical to it. The generators are numpy's own streams, bit for
//! bit. The API follows indrajala-ml's needs; there is no stability promise before 1.0.
//!
//! ```
//! use indrajala_math::linalg::matmul;
//! use indrajala_math::Array;
//!
//! let w = Array::from_matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
//! let x = Array::from_vector(vec![1.0, 0.0, -1.0]);
//! let y = matmul(&w, &x)?;
//! assert_eq!(y.data(), &[-2.0, -2.0]);
//! # Ok::<(), indrajala_math::Error>(())
//! ```

mod array;
pub mod attention;
pub mod batch_norm;
pub mod conv;
pub mod cr_exp;
mod error;
pub mod factored;
pub mod fused;
pub mod generator;
pub mod layer_norm;
pub mod linalg;
pub mod mnist;
pub mod ops;
pub mod program;
pub mod random;
pub mod recurrent;
pub mod scalar;
pub mod tokens;
pub mod ufuncs;

pub use array::{Array, Shape};
pub use error::Error;
