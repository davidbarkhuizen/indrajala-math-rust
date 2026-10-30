//! The crate's global generator, numpy's legacy `np.random` MT19937 bit for bit: `seed`, `random`,
//! `uniform` and the dropout mask.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::array::{Array, Shape};
use crate::Error;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// MT19937 (Matsumoto & Nishimura 1998) exactly as numpy's legacy `np.random` runs it: the same
/// seeding (`init_genrand`, `init_by_array`, and the `None` entropy procedure), twist, tempering
/// and `random_double`, so after `seed(s)` this crate's draws are bit-identical to numpy's after
/// `np.random.seed(s)`. numpy freezes this stream (NEP 19), so it is a stable target.
pub(crate) struct Mt19937 {
    key: [u32; N],
    pos: usize,
}

impl Mt19937 {
    /// numpy's `mt19937_seed`: seeding from one 32-bit word.
    fn init_genrand(seed: u32) -> Self {
        let mut key = [0u32; N];
        key[0] = seed;
        for i in 1..N {
            let previous = key[i - 1];
            key[i] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { key, pos: N }
    }

    /// numpy's `mt19937_init_by_array`: seeding from a non-empty key of any length.
    fn init_by_array(init_key: &[u32]) -> Self {
        let mut mt = Mt19937::init_genrand(19_650_218);
        let key = &mut mt.key;
        let (mut i, mut j) = (1, 0);
        for _ in 0..N.max(init_key.len()) {
            let previous = key[i - 1];
            key[i] = (key[i] ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(init_key[j])
                .wrapping_add(j as u32);
            i += 1;
            j = (j + 1) % init_key.len();
            if i >= N {
                key[0] = key[N - 1];
                i = 1;
            }
        }
        for _ in 0..N - 1 {
            let previous = key[i - 1];
            key[i] = (key[i] ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                key[0] = key[N - 1];
                i = 1;
            }
        }
        key[0] = UPPER_MASK; // non-zero initial array
        mt
    }

    /// numpy's unseeded state (the global `RandomState` built at import): key word 0 is
    /// `0x80000000` (so the state is never all zero), the rest OS entropy, and the position 623.
    fn from_entropy() -> Self {
        Mt19937 {
            key: entropy_key(),
            pos: N - 1,
        }
    }

    fn twist(&mut self) {
        let key = &mut self.key;
        let mix = |upper: u32, lower: u32| {
            let y = (upper & UPPER_MASK) | (lower & LOWER_MASK);
            (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 }
        };
        for i in 0..N - M {
            key[i] = key[i + M] ^ mix(key[i], key[i + 1]);
        }
        for i in N - M..N - 1 {
            key[i] = key[i + M - N] ^ mix(key[i], key[i + 1]);
        }
        key[N - 1] = key[M - 1] ^ mix(key[N - 1], key[0]);
        self.pos = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.pos >= N {
            self.twist();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

/// A generator's doubles in `[0, 1)`: this file's MT19937 and `generator.rs`'s PCG64.
pub trait DoubleSource {
    /// The next double in `[0, 1)`.
    fn next_double(&mut self) -> f64;
}

impl DoubleSource for Mt19937 {
    /// numpy's `random_double`: 27 + 26 bits of two draws, 53 random mantissa bits in `[0, 1)`.
    fn next_double(&mut self) -> f64 {
        let a = self.next_u32() >> 5;
        let b = self.next_u32() >> 6;
        (a as f64 * 67_108_864.0 + b as f64) / 9_007_199_254_740_992.0
    }
}

/// 623 words of OS entropy behind numpy's word-0 rule, for `seed(None)` and the unseeded state.
/// `RandomState` seeds its keys from the OS once per thread; the pid and the clock are mixed in
/// too, so processes forked after that still draw different words, as numpy's `SeedSequence` does.
fn entropy_key() -> [u32; N] {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let mut key = [0u32; N];
    for (i, pair) in key.chunks_mut(2).enumerate() {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u32(std::process::id());
        hasher.write_u128(nanos);
        hasher.write_usize(i);
        let word = hasher.finish();
        pair[0] = word as u32;
        pair[1] = (word >> 32) as u32;
    }
    key[0] = UPPER_MASK;
    key
}

/// The crate's one generator, the counterpart of numpy's global `RandomState`, separate from it.
/// Every draw happens in one call on the calling thread, in C order, never in the kernels' worker
/// threads, so the order is part of the contract. The crate never releases the GIL, so the lock
/// is never contended.
static STATE: Mutex<Option<Mt19937>> = Mutex::new(None);

fn locked_state() -> MutexGuard<'static, Option<Mt19937>> {
    STATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Runs `draw` on the global generator, seeding it from entropy first if nothing has yet.
fn with_rng<R>(draw: impl FnOnce(&mut Mt19937) -> R) -> R {
    let mut state = locked_state();
    draw(state.get_or_insert_with(Mt19937::from_entropy))
}

/// Seeds the global generator from entropy at import, as numpy seeds its global `RandomState`,
/// so processes forked from an importer inherit one stream, as they inherit numpy's.
pub fn seed_at_import() {
    with_rng(|_| ());
}

/// `seed(None)`: a fresh entropy key, keeping the position, as numpy's does (or the first state).
pub fn reseed_from_entropy() {
    let mut state = locked_state();
    match state.as_mut() {
        Some(mt) => mt.key = entropy_key(),
        None => *state = Some(Mt19937::from_entropy()),
    }
}

/// `seed(word)`: numpy's `init_genrand` for an integer seed.
pub fn seed_word(word: u32) {
    *locked_state() = Some(Mt19937::init_genrand(word));
}

/// `seed(sequence)`: numpy's `init_by_array` for a sequence's words.
pub fn seed_key(key: &[u32]) {
    *locked_state() = Some(Mt19937::init_by_array(key));
}

/// `data` in `shape`.
pub fn shaped(data: Vec<f64>, shape: Shape) -> Array {
    match shape {
        Shape::Vector(_) => Array::from_vector(data),
        Shape::Matrix(rows, cols) => Array::from_matrix(data, rows, cols),
    }
}

/// `np.random.random(shape)`: uniform draws in `[0, 1)`, filled in C order.
pub fn random(shape: Shape) -> Array {
    let data = with_rng(|rng| (0..shape.size()).map(|_| rng.next_double()).collect());
    shaped(data, shape)
}

/// `np.random.uniform(low, high, size=shape)`, `randomize()`'s draw: `low + range * u` with
/// `range = high - low` computed once, filled in C order. A non-finite range is numpy's
/// `OverflowError`.
pub fn uniform(low: f64, high: f64, shape: Shape) -> Result<Array, Error> {
    let range = high - low;
    if !range.is_finite() {
        return Err(Error::Overflow("Range exceeds valid bounds".to_string()));
    }
    let data = with_rng(|rng| (0..shape.size()).map(|_| low + range * rng.next_double()).collect());
    Ok(shaped(data, shape))
}

/// `(np.random.random(size) >= drop_probability)` as floats (1.0 kept, 0.0 dropped), the mask
/// `DropoutArrayLayer` draws. `fused.rs`'s `layer_dropout_forward*` draw it inside the same Rust
/// call as the matmul and sigmoid, so it is a plain `Vec<f64>`; `bernoulli_mask` below is its
/// Python-visible form.
pub fn draw_bernoulli_mask(drop_probability: f64, size: usize) -> Vec<f64> {
    with_rng(|rng| bernoulli_values(rng, drop_probability, size))
}

/// `size` mask entries drawn from `rng`, one double each: 1.0 where it is `>= drop_probability`.
pub fn bernoulli_values(rng: &mut impl DoubleSource, drop_probability: f64, size: usize) -> Vec<f64> {
    (0..size)
        .map(|_| {
            if rng.next_double() >= drop_probability {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// `(np.random.random(shape) >= drop_probability).astype(float)`: `draw_bernoulli_mask` as an
/// array, so the mask can be checked against numpy on its own.
pub fn bernoulli_mask(drop_probability: f64, shape: Shape) -> Array {
    shaped(draw_bernoulli_mask(drop_probability, shape.size()), shape)
}
