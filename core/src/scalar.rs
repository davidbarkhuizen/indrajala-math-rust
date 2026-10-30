//! The crate's scalar rules where IEEE 754 or the compiler would leave the bits to the machine
//! (the aarch64 workplan's D6, in indrajala-ml), so a Rust-backend network computes the same
//! bits on every architecture. `exp` is `cr_exp`, correctly rounded (D4).
//!
//! What stays the machine's: the sign and payload of a NaN that an invalid operation creates
//! (`inf - inf`, `0 * inf`): x86 makes it negative, ARM positive. The cross-machine claim is
//! for finite data, where no such NaN arises; a NaN already in the data propagates as below.

/// ReLU: `v` above zero, NaN propagated (as numpy's `maximum`), and `+0.0` otherwise, so `-0.0`
/// and `+0.0` both give `+0.0` (Rust's `f64::max` leaves which zero to the hardware, x86's
/// `maxsd` and ARM's `fmaxnm` differ, and drops a NaN).
#[inline(always)]
pub fn relu(v: f64) -> f64 {
    if v > 0.0 || v.is_nan() {
        v
    } else {
        0.0
    }
}

/// The larger of a running maximum and `v`, by relu's comparison: a left fold of it from
/// `f64::NEG_INFINITY` keeps the first of equal values (so `-0.0` before `+0.0` stays `-0.0`)
/// and propagates a NaN (the row's last).
#[inline(always)]
pub fn max(running: f64, v: f64) -> f64 {
    if v > running || v.is_nan() {
        v
    } else {
        running
    }
}

/// `base^exponent` by square-and-multiply from the exponent's low bit, the order of
/// compiler-rt's `__powidf2`, which `f64::powi` usually but not by contract lowers to (LLVM may
/// expand it otherwise); written out so the order is the crate's. Adam's bias corrections.
#[inline]
pub fn powi(base: f64, exponent: u32) -> f64 {
    let (mut a, mut b, mut r) = (base, exponent, 1.0);
    loop {
        if b & 1 == 1 {
            r *= a;
        }
        b >>= 1;
        if b == 0 {
            return r;
        }
        a *= a;
    }
}
