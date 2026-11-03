//! Elementwise ops over [`Array`](crate::Array)s: arithmetic, functions of one array, and the masks
//! the backward passes apply.

pub mod arith;
pub mod mask;
pub mod unary;
