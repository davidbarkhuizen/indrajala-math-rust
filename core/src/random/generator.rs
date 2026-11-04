//! numpy's `default_rng`: `Generator(PCG64(SeedSequence(seed)))`, bit for bit. `SeedSequence`
//! hashes a seed into a 128-bit pool and expands it into PCG64's 256-bit seed; `Pcg64` is
//! O'Neill's PCG XSL-RR 128/64 as numpy runs it; `Generator` draws numpy's doubles from it. A
//! separate stream from [`legacy`](super::legacy)'s global MT19937 (numpy's legacy `np.random`), as numpy keeps
//! its `Generator` apart from its legacy functions. Each `Generator` owns its state, which reads
//! and writes as numpy's `bit_generator.state`, so a state moves between numpy and the crate.

use crate::array::{Array, Shape};
use crate::random::legacy::{shaped, DoubleSource};
use crate::Error;

/// numpy's default pool: four 32-bit words.
pub const POOL_SIZE: usize = 4;
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

/// `PCG_DEFAULT_MULTIPLIER_128`.
const PCG_MULTIPLIER: u128 = (2_549_297_995_355_413_924u128 << 64) | 4_865_540_595_714_422_341u128;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut value = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let result = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^ (result >> XSHIFT)
}

/// `SeedSequence.mix_entropy`: hashes the assembled entropy words into the pool.
pub(crate) fn mix_entropy(entropy: &[u32]) -> [u32; POOL_SIZE] {
    let mut hash_const = INIT_A;
    let mut pool = [0u32; POOL_SIZE];
    for (i, word) in pool.iter_mut().enumerate() {
        *word = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hash_const);
    }
    for i_src in 0..POOL_SIZE {
        for i_dst in 0..POOL_SIZE {
            if i_src != i_dst {
                pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src], &mut hash_const));
            }
        }
    }
    for &word in entropy.iter().skip(POOL_SIZE) {
        for i_dst in 0..POOL_SIZE {
            pool[i_dst] = mix(pool[i_dst], hashmix(word, &mut hash_const));
        }
    }
    pool
}

/// `SeedSequence.generate_state(n_words, np.uint32)`: the pool expanded into `n_words` words.
pub(crate) fn generate_words(pool: &[u32; POOL_SIZE], n_words: usize) -> Vec<u32> {
    let mut hash_const = INIT_B;
    (0..n_words)
        .map(|i| {
            let mut value = pool[i % POOL_SIZE] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            value = value.wrapping_mul(hash_const);
            value ^ (value >> XSHIFT)
        })
        .collect()
}

/// numpy's uint64 view of the uint32 words: little-endian pairs, low word first.
pub(crate) fn words_to_u64(words: &[u32]) -> Vec<u64> {
    words
        .chunks(2)
        .map(|pair| pair[0] as u64 | (pair[1] as u64) << 32)
        .collect()
}

/// `np.random.SeedSequence`'s pool, mixed from the entropy's words and the spawn key's; the Python
/// bindings read a seed's Python forms into these words, as numpy does.
#[derive(Clone, Debug)]
pub struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    /// numpy's `get_assembled_entropy` then `mix_entropy`: a spawn key pads the run entropy to the
    /// pool size with zeros, so the spawn key's words never stand in for missing entropy words.
    pub fn from_words(entropy_words: &[u32], spawn_words: &[u32]) -> Self {
        let mut assembled = entropy_words.to_vec();
        if !spawn_words.is_empty() && assembled.len() < POOL_SIZE {
            assembled.resize(POOL_SIZE, 0);
        }
        assembled.extend_from_slice(spawn_words);
        SeedSequence {
            pool: mix_entropy(&assembled),
        }
    }

    /// The four mixed 32-bit words.
    pub fn pool(&self) -> [u32; POOL_SIZE] {
        self.pool
    }

    /// `generate_state(n_words, np.uint32)`.
    pub fn generate_state_u32(&self, n_words: usize) -> Vec<u32> {
        generate_words(&self.pool, n_words)
    }

    /// `generate_state(n_words, np.uint64)`: each word two uint32 words, low first.
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        words_to_u64(&generate_words(&self.pool, 2 * n_words))
    }

    /// The 256 bits PCG64 seeds from: `generate_state(4, np.uint64)`.
    pub(crate) fn pcg64(&self) -> Pcg64 {
        let words = words_to_u64(&generate_words(&self.pool, 8));
        Pcg64::seeded(
            (words[0] as u128) << 64 | words[1] as u128,
            (words[2] as u128) << 64 | words[3] as u128,
        )
    }
}

/// PCG XSL-RR 128/64 (O'Neill 2014) as numpy's `PCG64`: a 128-bit LCG stepped before each
/// output, whose two halves are xored and rotated by the state's top six bits.
#[derive(Clone, Debug)]
pub(crate) struct Pcg64 {
    /// The LCG's state.
    pub state: u128,
    /// Its increment (odd).
    pub inc: u128,
}

impl Pcg64 {
    /// `pcg64_set_seed`: `pcg_setseq_128_srandom_r(initstate, initseq)`.
    pub fn seeded(initstate: u128, initseq: u128) -> Self {
        let mut pcg = Pcg64 {
            state: 0,
            inc: initseq << 1 | 1,
        };
        pcg.step();
        pcg.state = pcg.state.wrapping_add(initstate);
        pcg.step();
        pcg
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULTIPLIER).wrapping_add(self.inc);
    }

    /// The next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let folded = (self.state >> 64) as u64 ^ self.state as u64;
        folded.rotate_right((self.state >> 122) as u32)
    }
}

impl DoubleSource for Pcg64 {
    /// numpy's `next_double`: the top 53 bits of one draw, in `[0, 1)`.
    fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

/// The smallest 53-bit draw `n` with `n / 2^53 >= p`. Scaling by a power of two is exact, and so is
/// `ceil`, whose result the cast saturates: a negative `p` gives 0, every draw at or above it, and
/// `p >= 1` gives a value no draw reaches. No draw compares `>=` a NaN.
fn bits_threshold(p: f64) -> u64 {
    if p.is_nan() {
        u64::MAX
    } else {
        (p * 9_007_199_254_740_992.0).ceil() as u64
    }
}

/// numpy's `bit_generator.state` for PCG64, as plain fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeneratorState {
    /// PCG64's state.
    pub state: u128,
    /// PCG64's increment.
    pub inc: u128,
    /// numpy's buffered half of a 64-bit draw for 32-bit draws, which nothing here makes; kept so
    /// a state round-trips unchanged.
    pub has_uint32: i32,
    /// The buffered half.
    pub uinteger: u32,
}

/// `np.random.default_rng(seed)`'s `Generator`: `random`, `uniform` and the dropout mask, drawn
/// from its own PCG64 in C order on the calling thread.
#[derive(Clone, Debug)]
pub struct Generator {
    pcg: Pcg64,
    has_uint32: i32,
    uinteger: u32,
}

impl Generator {
    /// `Generator(PCG64(seed_sequence))`.
    pub fn from_seed_sequence(seed_sequence: &SeedSequence) -> Self {
        Generator {
            pcg: seed_sequence.pcg64(),
            has_uint32: 0,
            uinteger: 0,
        }
    }

    /// The dropout mask the fused `layer_dropout_forward*` draw with this generator.
    ///
    /// Each entry is [`bernoulli_values`](crate::random::legacy::bernoulli_values)'s `next_double() >= drop_probability`, compared on the
    /// draw's 53 bits: `n / 2^53 >= p` is exactly `n >= ceil(p * 2^53)`, and the conversion and float
    /// compare were a large part of an attention layer's own time (the crate performance workplan,
    /// stage 7).
    pub fn draw_bernoulli_mask(&mut self, drop_probability: f64, size: usize) -> Vec<f64> {
        let threshold = bits_threshold(drop_probability);
        (0..size)
            .map(|_| {
                if self.pcg.next_u64() >> 11 >= threshold {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// `rng.random(shape)`: doubles in `[0, 1)`, filled in C order.
    pub fn random(&mut self, shape: Shape) -> Array {
        let data = (0..shape.size()).map(|_| self.pcg.next_double()).collect();
        shaped(data, shape)
    }

    /// `rng.uniform(low, high, shape)`: `low + (high - low) * u`, filled in C order. A non-finite
    /// range is numpy's `OverflowError` and a negative one its `ValueError`, unlike the legacy
    /// `uniform`, which accepts `high < low`.
    pub fn uniform(&mut self, low: f64, high: f64, shape: Shape) -> Result<Array, Error> {
        let range = high - low;
        if !range.is_finite() {
            return Err(Error::Overflow("high - low range exceeds valid bounds".to_string()));
        }
        if range < 0.0 {
            return Err(Error::invalid("high - low < 0"));
        }
        let data = (0..shape.size())
            .map(|_| low + range * self.pcg.next_double())
            .collect();
        Ok(shaped(data, shape))
    }

    /// `(rng.random(shape) >= drop_probability).astype(float)`: 1.0 kept, 0.0 dropped.
    pub fn bernoulli_mask(&mut self, drop_probability: f64, shape: Shape) -> Array {
        shaped(self.draw_bernoulli_mask(drop_probability, shape.size()), shape)
    }

    /// numpy's `bit_generator.state`.
    pub fn state(&self) -> GeneratorState {
        GeneratorState {
            state: self.pcg.state,
            inc: self.pcg.inc,
            has_uint32: self.has_uint32,
            uinteger: self.uinteger,
        }
    }

    /// Sets numpy's `bit_generator.state`.
    pub fn set_state(&mut self, state: GeneratorState) {
        self.pcg = Pcg64 {
            state: state.state,
            inc: state.inc,
        };
        self.has_uint32 = state.has_uint32;
        self.uinteger = state.uinteger;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::legacy::bernoulli_values;

    /// The integer compare against the float compare it replaces, draw for draw and with the
    /// generator left in the same state, at probabilities on and beside a draw's own values.
    #[test]
    fn mask_matches_the_float_compare() {
        const TWO_53: f64 = 9_007_199_254_740_992.0;
        let mut probabilities = vec![
            0.0,
            -0.0,
            0.1,
            0.5,
            0.9,
            1.0,
            2.0,
            -1.0,
            f64::MIN_POSITIVE,
            5e-324,
            1.0 / TWO_53,
            1.5 / TWO_53,
            1.0 - 1.0 / TWO_53,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        // a draw's exact value as the probability, and the floats either side of it
        let generator = || Generator::from_seed_sequence(&SeedSequence::from_words(&[11], &[]));
        let mut source = generator();
        for _ in 0..64 {
            let u = source.pcg.next_double();
            probabilities.extend([u, u.next_down(), u.next_up()]);
        }
        for p in probabilities {
            let (mut a, mut b) = (generator(), generator());
            let mask = a.draw_bernoulli_mask(p, 256);
            let expected = bernoulli_values(&mut b.pcg, p, 256);
            assert_eq!(mask, expected, "probability {p}");
            assert_eq!(a.pcg.next_u64(), b.pcg.next_u64(), "probability {p}");
        }
    }
}
