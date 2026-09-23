use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use pyo3::prelude::*;

use crate::array::{parse_shape, RustArray, Shape};

/// A hand-rolled xorshift128+ generator - a small, well-known public-domain algorithm,
/// implemented directly rather than pulled from the `rand` crate, matching this crate's
/// "hand-build everything, no convenience dependencies" posture.
///
/// This is the one operation in the whole interface subset where parity with numpy cannot mean
/// bit-identical output - a hand-rolled generator can never reproduce numpy's Mersenne Twister
/// stream, seeded or not. `randomize()`'s own "same seed -> same trained weights" regression gate
/// breaks once this backs production, even though every other operation in this core matches
/// numpy exactly - not a bug, a fact worth stating plainly.
struct Xorshift128Plus {
    state0: u64,
    state1: u64,
}

impl Xorshift128Plus {
    fn new(seed: u64) -> Self {
        // splitmix64 expands a single seed word into two well-mixed 64-bit words, avoiding the
        // all-zero state xorshift128+ can never recover from.
        let mut splitmix_state = seed;
        let mut next_splitmix = || {
            splitmix_state = splitmix_state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = splitmix_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        Xorshift128Plus {
            state0: next_splitmix(),
            state1: next_splitmix(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut s1 = self.state0;
        let s0 = self.state1;
        self.state0 = s0;
        s1 ^= s1 << 23;
        s1 ^= s1 >> 17;
        s1 ^= s0 ^ (s0 >> 26);
        self.state1 = s1;
        self.state1.wrapping_add(self.state0)
    }

    /// A uniform draw in `[0.0, 1.0)` - the standard "top 53 bits become the mantissa" technique
    /// for turning a random 64-bit word into a random `f64`.
    fn next_unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Seeds each `uniform()` call freshly from wall-clock time plus a monotonically increasing
/// counter (guarding against two calls landing in the same clock tick) - there is no
/// numpy-seed-compatible RNG to match (see this module's own doc comment), so no seed parameter
/// is exposed; every call simply draws fresh randomness, matching `np.random.uniform`'s own
/// reliance on numpy's global, unseeded-by-default RNG state.
fn fresh_seed() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ counter.wrapping_mul(0x9E3779B97F4A7C15)
}

/// Fills an array of the given shape with independent uniform draws in `[low, high)` -
/// `randomize()`'s own `np.random.uniform(-limit, limit, size=shape)`.
#[pyfunction]
pub fn uniform(low: f64, high: f64, shape: &PyAny) -> PyResult<RustArray> {
    let shape = parse_shape(shape)?;
    let mut rng = Xorshift128Plus::new(fresh_seed());
    let data: Vec<f64> = (0..shape.size())
        .map(|_| low + (high - low) * rng.next_unit_f64())
        .collect();
    Ok(match shape {
        Shape::Vector(_) => RustArray::from_vector(data),
        Shape::Matrix(rows, cols) => RustArray::from_matrix(data, rows, cols),
    })
}

/// Draws `size` independent inverted-dropout keep/drop entries (1.0 kept, 0.0 dropped), each
/// `>= drop_probability` against a fresh uniform-in-`[0,1)` draw - exactly
/// `np.random.random(shape) >= drop_probability`'s own comparison (`DropoutArrayLayer.forward`),
/// just inlined here as a flat `Vec<f64>` rather than a `RustArray` so `fused.rs`'s
/// `layer_dropout_forward`/`layer_dropout_forward_batch` can draw a mask internally, in the same
/// Rust call that also does the matmul/sigmoid, without a second Python/Rust FFI crossing - the
/// same "one Rust call per layer method" discipline `layer_forward`/`layer_relu_forward` follow.
/// Kept `pub(crate)` (not a `#[pyfunction]` itself) since `bernoulli_mask` below is the
/// Python-visible, independently-testable entry point to this same logic.
pub(crate) fn draw_bernoulli_mask(drop_probability: f64, size: usize) -> Vec<f64> {
    let mut rng = Xorshift128Plus::new(fresh_seed());
    (0..size)
        .map(|_| if rng.next_unit_f64() >= drop_probability { 1.0 } else { 0.0 })
        .collect()
}

/// The standalone, Python-visible counterpart to `draw_bernoulli_mask` above - lets the mask
/// distribution itself be checked directly (statistically, since bit-identical parity with numpy
/// isn't achievable here - see this module's own doc comment), independently of
/// `DropoutArrayLayer`'s Rust counterpart or its fused forward ops, matching how `array_relu`/
/// `array_softmax` (`ufuncs.rs`) are each validated against numpy on their own.
#[pyfunction]
pub fn bernoulli_mask(drop_probability: f64, shape: &PyAny) -> PyResult<RustArray> {
    let shape = parse_shape(shape)?;
    let data = draw_bernoulli_mask(drop_probability, shape.size());
    Ok(match shape {
        Shape::Vector(_) => RustArray::from_vector(data),
        Shape::Matrix(rows, cols) => RustArray::from_matrix(data, rows, cols),
    })
}
