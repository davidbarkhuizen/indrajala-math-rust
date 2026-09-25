use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::array::{parse_shape, RustArray, Shape};

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;
const SEED_RANGE_MESSAGE: &str = "Seed must be between 0 and 2**32 - 1";

/// MT19937 (Matsumoto & Nishimura 1998) exactly as numpy's legacy `np.random` runs it: the same
/// seeding (`init_genrand`, `init_by_array`, and the `None` entropy procedure), twist, tempering
/// and `random_double`, so after `seed(s)` this crate's draws are bit-identical to numpy's after
/// `np.random.seed(s)`. numpy freezes this stream (NEP 19), so it is a stable target.
struct Mt19937 {
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
pub(crate) fn seed_at_import() {
    with_rng(|_| ());
}

/// What `np.random.seed`'s sequence path makes of a seed: its dimensions and, per element,
/// the value if the element survives numpy's `astype(np.int64, casting='safe')`.
#[derive(Default)]
struct DiscoveredSeed {
    dims: Vec<usize>,
    ndim: Option<usize>,
    words: Vec<Option<i64>>,
}

fn inhomogeneous(depth: usize) -> PyErr {
    PyValueError::new_err(format!(
        "setting an array element with a sequence. The requested array has an inhomogeneous \
         shape after {depth} dimensions."
    ))
}

impl DiscoveredSeed {
    fn leaf(&mut self, depth: usize, word: Option<i64>) -> PyResult<()> {
        match self.ndim {
            None if self.dims.len() == depth => self.ndim = Some(depth),
            Some(ndim) if ndim == depth => {}
            _ => return Err(inhomogeneous(depth)),
        }
        self.words.push(word);
        Ok(())
    }

    fn sequence(&mut self, depth: usize, items: Vec<&PyAny>, safe_kind: Option<bool>) -> PyResult<()> {
        if self.ndim.is_some_and(|ndim| depth >= ndim) {
            return Err(inhomogeneous(depth));
        }
        match self.dims.get(depth) {
            Some(&len) if len != items.len() => return Err(inhomogeneous(depth)),
            Some(_) => {}
            None => self.dims.push(items.len()),
        }
        for item in items {
            self.discover(item, depth + 1, safe_kind)?;
        }
        Ok(())
    }

    /// Walks a seed the way `np.asarray` discovers its shape and dtype. `safe_kind` is set inside
    /// a buffer, whose format fixes the dtype of every element (`Some(false)`: none survive).
    fn discover(&mut self, obj: &PyAny, depth: usize, safe_kind: Option<bool>) -> PyResult<()> {
        let py = obj.py();
        let is_text = obj.is_instance_of::<pyo3::types::PyString>() || obj.is_instance_of::<pyo3::types::PyBytes>();
        if is_text {
            return self.leaf(depth, None);
        }
        if obj.hasattr("dtype")? && obj.hasattr("ndim")? {
            // a numpy array or scalar
            let dtype = obj.getattr("dtype")?;
            let kind: char = dtype.getattr("kind")?.extract()?;
            let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
            let safe = matches!(kind, 'b' | 'i') || (kind == 'u' && itemsize < 8);
            return if obj.getattr("ndim")?.extract::<usize>()? == 0 {
                let word = if safe {
                    Some(obj.call_method0("item")?.extract()?)
                } else {
                    None
                };
                self.leaf(depth, word)
            } else {
                self.sequence(depth, obj.iter()?.collect::<PyResult<_>>()?, Some(safe))
            };
        }
        if safe_kind.is_none() && unsafe { pyo3::ffi::PyObject_CheckBuffer(obj.as_ptr()) } == 1 {
            // array.array, bytearray, memoryview: the buffer's format is the dtype
            let view = py.import("builtins")?.getattr("memoryview")?.call1((obj,))?;
            let format: String = view.getattr("format")?.extract()?;
            let code = format.trim_start_matches(['@', '=', '<', '>', '!']);
            let itemsize: usize = view.getattr("itemsize")?.extract()?;
            let safe = matches!(code, "?" | "b" | "h" | "i" | "l" | "q" | "n")
                || (matches!(code, "B" | "H" | "I" | "L" | "Q" | "N") && itemsize < 8);
            let as_list = view.call_method0("tolist")?;
            return self.discover(as_list, depth, Some(safe));
        }
        if unsafe { pyo3::ffi::PySequence_Check(obj.as_ptr()) } == 1 {
            return self.sequence(depth, obj.iter()?.collect::<PyResult<_>>()?, safe_kind);
        }
        // a Python int within int64 is cast safely; anything else numpy holds as a float,
        // complex or object dtype, and none of those cast
        let word = if safe_kind != Some(false) && obj.is_instance_of::<pyo3::types::PyLong>() {
            obj.extract::<i64>().ok()
        } else {
            None
        };
        self.leaf(depth, word)
    }
}

/// `np.random.seed`'s sequence path: `np.asarray(seed)`, non-empty, cast to int64 with
/// `casting='safe'`, 1-D, every word in `[0, 2**32 - 1]` - checked in that order, so each
/// rejection raises numpy's exception type.
fn key_from_sequence(seed: &PyAny) -> PyResult<Vec<u32>> {
    let mut discovered = DiscoveredSeed::default();
    discovered.discover(seed, 0, None)?;
    if discovered.words.is_empty() {
        return Err(PyValueError::new_err("Seed must be non-empty"));
    }
    let words: Vec<i64> = discovered
        .words
        .into_iter()
        .collect::<Option<_>>()
        .ok_or_else(|| PyTypeError::new_err("Cannot cast the seed to int64 according to the rule 'safe'"))?;
    if discovered.ndim != Some(1) {
        return Err(PyValueError::new_err("Seed array must be 1-d"));
    }
    words
        .into_iter()
        .map(|word| u32::try_from(word).map_err(|_| PyValueError::new_err(SEED_RANGE_MESSAGE)))
        .collect()
}

/// `np.random.seed(seed)` for this crate's generator, which is separate from numpy's: after
/// `seed(s)` it draws what numpy draws after `np.random.seed(s)`, for every seed numpy accepts,
/// and raises numpy's exception type for every seed it rejects. An integer (anything
/// `operator.index` takes, after `squeeze` if the seed has one) runs `init_genrand`; any other
/// sequence runs `init_by_array`; `None` reseeds from entropy and keeps the position, as numpy's
/// does.
#[pyfunction]
#[pyo3(signature = (seed=None))]
pub fn seed(py: Python<'_>, seed: Option<&PyAny>) -> PyResult<()> {
    let Some(mut seed) = seed else {
        let mut state = locked_state();
        match state.as_mut() {
            Some(mt) => mt.key = entropy_key(),
            None => *state = Some(Mt19937::from_entropy()),
        }
        return Ok(());
    };
    if seed.hasattr("squeeze")? {
        seed = seed.call_method0("squeeze")?;
    }
    let seeded = match py.import("operator")?.getattr("index")?.call1((seed,)) {
        Ok(index) => {
            let word: u32 = index.extract().map_err(|_| PyValueError::new_err(SEED_RANGE_MESSAGE))?;
            Mt19937::init_genrand(word)
        }
        Err(error) if error.is_instance_of::<PyTypeError>(py) => Mt19937::init_by_array(&key_from_sequence(seed)?),
        Err(error) => return Err(error),
    };
    *locked_state() = Some(seeded);
    Ok(())
}

fn shaped(data: Vec<f64>, shape: Shape) -> RustArray {
    match shape {
        Shape::Vector(_) => RustArray::from_vector(data),
        Shape::Matrix(rows, cols) => RustArray::from_matrix(data, rows, cols),
    }
}

/// `np.random.random(shape)`: uniform draws in `[0, 1)`, filled in C order.
#[pyfunction]
pub fn random(shape: &PyAny) -> PyResult<RustArray> {
    let shape = parse_shape(shape)?;
    let data = with_rng(|rng| (0..shape.size()).map(|_| rng.next_double()).collect());
    Ok(shaped(data, shape))
}

/// `np.random.uniform(low, high, size=shape)`, `randomize()`'s draw: `low + range * u` with
/// `range = high - low` computed once, filled in C order. A non-finite range raises numpy's
/// `OverflowError`.
#[pyfunction]
pub fn uniform(low: f64, high: f64, shape: &PyAny) -> PyResult<RustArray> {
    let shape = parse_shape(shape)?;
    let range = high - low;
    if !range.is_finite() {
        return Err(PyOverflowError::new_err("Range exceeds valid bounds"));
    }
    let data = with_rng(|rng| (0..shape.size()).map(|_| low + range * rng.next_double()).collect());
    Ok(shaped(data, shape))
}

/// `(np.random.random(size) >= drop_probability)` as floats (1.0 kept, 0.0 dropped), the mask
/// `DropoutArrayLayer` draws. `fused.rs`'s `layer_dropout_forward*` draw it inside the same Rust
/// call as the matmul and sigmoid, so it is a plain `Vec<f64>`; `bernoulli_mask` below is its
/// Python-visible form.
pub(crate) fn draw_bernoulli_mask(drop_probability: f64, size: usize) -> Vec<f64> {
    with_rng(|rng| {
        (0..size)
            .map(|_| {
                if rng.next_double() >= drop_probability {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    })
}

/// `(np.random.random(shape) >= drop_probability).astype(float)`: `draw_bernoulli_mask` as an
/// array, so the mask can be checked against numpy on its own.
#[pyfunction]
pub fn bernoulli_mask(drop_probability: f64, shape: &PyAny) -> PyResult<RustArray> {
    let shape = parse_shape(shape)?;
    Ok(shaped(draw_bernoulli_mask(drop_probability, shape.size()), shape))
}
