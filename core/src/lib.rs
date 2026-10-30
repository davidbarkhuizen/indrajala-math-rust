//! The array type and kernels behind [indrajala-math-rust](https://pypi.org/project/indrajala-math-rust/),
//! indrajala-ml's backend, as a pure-Rust library. Every operation borrows its operands
//! (`&Array`) and returns a new [`Array`]; bindings that hold an `Array` inside their own type copy
//! nothing at the boundary.

mod array;
mod error;
pub mod linalg;

pub use array::{transpose, Array, Shape};
pub use error::Error;
