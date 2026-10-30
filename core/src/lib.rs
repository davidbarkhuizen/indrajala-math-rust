//! The array type and kernels behind [indrajala-math-rust](https://pypi.org/project/indrajala-math-rust/),
//! indrajala-ml's backend, as a pure-Rust library. Every operation borrows its operands
//! (`&Array`) and returns a new [`Array`]; bindings that hold an `Array` inside their own type copy
//! nothing at the boundary.

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
