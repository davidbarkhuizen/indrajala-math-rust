//! The layers: one function per pass of an indrajala-ml layer class, the whole computation in one
//! call. They build on [`num`](crate::num), [`ops`](crate::ops), [`linalg`](crate::linalg) and
//! [`random`](crate::random), and on each other.

pub mod attention;
pub mod conv;
pub mod dense;
pub mod dropout;
pub mod gated;
pub mod norm;
pub mod optim;
pub mod recurrent;
pub mod softmax;
pub mod tokens;
