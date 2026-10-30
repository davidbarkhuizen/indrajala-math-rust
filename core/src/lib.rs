//! The array type and kernels behind [indrajala-math-rust](https://pypi.org/project/indrajala-math-rust/),
//! indrajala-ml's backend, as a pure-Rust library. Every operation reads borrowed arrays
//! ([`ArrayView`]) and returns an owned one ([`Array`]), so bindings over it copy nothing at the
//! boundary.

mod array;
mod error;
pub mod linalg;

pub use array::{transpose, Array, ArrayView, Shape};
pub use error::Error;
