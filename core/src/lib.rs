//! The array type and kernels behind [indrajala-math-rust](https://pypi.org/project/indrajala-math-rust/),
//! indrajala-ml's backend, as a pure-Rust library. Every operation borrows its operands
//! (`&Array`) and returns a new [`Array`]; bindings that hold an `Array` inside their own type copy
//! nothing at the boundary.

mod array;
pub mod batch_norm;
pub mod conv;
pub mod cr_exp;
mod error;
pub mod layer_norm;
pub mod linalg;
pub mod ops;
pub mod scalar;

pub use array::{Array, Shape};
pub use error::Error;
