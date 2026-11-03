//! Scalar `f64` rules, over values and slices, no [`Array`](crate::Array): the bottom layer, whose
//! bits every op above inherits.

pub mod activation;
pub mod erf;
pub mod exp;
pub mod scalar;
