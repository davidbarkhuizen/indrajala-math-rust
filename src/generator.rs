//! numpy's `default_rng`: `Generator(PCG64(SeedSequence(seed)))`, bit for bit. `SeedSequence`
//! hashes a seed into a 128-bit pool and expands it into PCG64's 256-bit seed; `Pcg64` is
//! O'Neill's PCG XSL-RR 128/64 as numpy runs it; `Generator` draws numpy's doubles from it. A
//! separate stream from `random.rs`'s global MT19937 (numpy's legacy `np.random`), as numpy keeps
//! its `Generator` apart from its legacy functions. Each `Generator` owns its state, which reads
//! and writes as numpy's `bit_generator.state`, so a state moves between numpy and the crate.

use std::sync::atomic::{AtomicUsize, Ordering};

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::array::{parse_shape, RustArray};
use crate::random::{bernoulli_values, shaped, DoubleSource};

const POOL_SIZE: usize = 4;
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
fn mix_entropy(entropy: &[u32]) -> [u32; POOL_SIZE] {
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
fn generate_words(pool: &[u32; POOL_SIZE], n_words: usize) -> Vec<u32> {
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
fn words_to_u64(words: &[u32]) -> Vec<u64> {
    words
        .chunks(2)
        .map(|pair| pair[0] as u64 | (pair[1] as u64) << 32)
        .collect()
}

/// `_int_to_uint32_array`: a non-negative integer as 32-bit words, lowest first (`[0]` for 0).
fn int_to_words(value: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    let py = value.py();
    let mut n = py.import("operator")?.getattr("index")?.call1((value,))?;
    if n.lt(0)? {
        return Err(PyValueError::new_err("expected non-negative integer"));
    }
    let mut words = vec![];
    loop {
        words.push(n.call_method1("__and__", (u32::MAX,))?.extract::<u32>()?);
        n = n.call_method1("__rshift__", (32,))?;
        if !n.is_truthy()? {
            return Ok(words);
        }
    }
}

/// numpy's classes, if numpy is imported: no object can be a numpy object otherwise, and the
/// crate doesn't import it.
struct NumpyTypes<'py> {
    ndarray: Bound<'py, PyAny>,
    integer: Bound<'py, PyAny>,
    inexact: Bound<'py, PyAny>,
}

fn numpy_types(py: Python<'_>) -> PyResult<Option<NumpyTypes<'_>>> {
    let modules = py.import("sys")?.getattr("modules")?;
    let Some(numpy) = modules
        .call_method1("get", ("numpy",))?
        .extract::<Option<Bound<'_, PyAny>>>()?
    else {
        return Ok(None);
    };
    Ok(Some(NumpyTypes {
        ndarray: numpy.getattr("ndarray")?,
        integer: numpy.getattr("integer")?,
        inexact: numpy.getattr("inexact")?,
    }))
}

fn is_instance(obj: &Bound<'_, PyAny>, numpy_type: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    numpy_type.map_or(Ok(false), |numpy_type| obj.is_instance(numpy_type))
}

/// `_coerce_to_uint32_array` for one element of an entropy sequence or spawn key: an integer, a
/// float (rejected) or a seed string. Anything else that has a length is a nested sequence; the
/// rest fail in `len()`, as in numpy.
fn element_words(item: &Bound<'_, PyAny>, numpy: Option<&NumpyTypes<'_>>) -> PyResult<Vec<u32>> {
    let py = item.py();
    if let Ok(text) = item.cast::<PyString>() {
        let text = text.to_str()?;
        let base = if text.starts_with("0x") {
            16
        } else if text.starts_with(|c: char| c.is_ascii_digit()) {
            10
        } else {
            return Err(PyValueError::new_err("unrecognized seed string"));
        };
        let value = py.import("builtins")?.getattr("int")?.call1((text, base))?;
        return int_to_words(&value);
    }
    if item.is_instance_of::<pyo3::types::PyInt>() || is_instance(item, numpy.map(|np| &np.integer))? {
        return int_to_words(item);
    }
    if item.is_instance_of::<pyo3::types::PyFloat>() || is_instance(item, numpy.map(|np| &np.inexact))? {
        return Err(PyTypeError::new_err("seed must be integer"));
    }
    let nested = || PyTypeError::new_err("SeedSequence does not accept nested sequences.");
    if item.hasattr("__len__")? {
        return Err(nested());
    }
    item.len()?;
    Err(nested())
}

/// `_coerce_to_uint32_array` for a whole entropy or spawn key: an integer's words, or every
/// element's words of a sequence, concatenated.
fn coerce_words(value: &Bound<'_, PyAny>, numpy: Option<&NumpyTypes<'_>>) -> PyResult<Vec<u32>> {
    if value.is_instance_of::<pyo3::types::PyInt>() || is_instance(value, numpy.map(|np| &np.integer))? {
        return int_to_words(value);
    }
    // numpy takes len() first, so a 0-d array fails there rather than in iteration
    value.len()?;
    let mut words = vec![];
    for item in value.try_iter()? {
        words.extend(element_words(&item?, numpy)?);
    }
    Ok(words)
}

/// `np.random.SeedSequence(entropy, spawn_key=())`, with numpy's default pool of four words.
#[pyclass(module = "indrajala_math_rust", frozen, skip_from_py_object)]
pub struct SeedSequence {
    entropy: Py<PyAny>,
    spawn_key: Py<PyTuple>,
    entropy_words: Vec<u32>,
    pool: [u32; POOL_SIZE],
    n_children_spawned: AtomicUsize,
}

impl SeedSequence {
    fn from_parts(
        entropy: Py<PyAny>,
        entropy_words: Vec<u32>,
        spawn_key: Bound<'_, PyTuple>,
        numpy: Option<&NumpyTypes<'_>>,
    ) -> PyResult<Self> {
        // get_assembled_entropy: a spawn key pads the run entropy to the pool size with zeros,
        // so the spawn key's words never stand in for missing entropy words
        let spawn_words = coerce_words(spawn_key.as_any(), numpy)?;
        let mut assembled = entropy_words.clone();
        if !spawn_words.is_empty() && assembled.len() < POOL_SIZE {
            assembled.resize(POOL_SIZE, 0);
        }
        assembled.extend(spawn_words);
        Ok(SeedSequence {
            entropy,
            spawn_key: spawn_key.unbind(),
            entropy_words,
            pool: mix_entropy(&assembled),
            n_children_spawned: AtomicUsize::new(0),
        })
    }

    /// The 256 bits PCG64 seeds from: `generate_state(4, np.uint64)`.
    fn pcg64_seed(&self) -> Pcg64 {
        let words = words_to_u64(&generate_words(&self.pool, 8));
        Pcg64::seeded(
            (words[0] as u128) << 64 | words[1] as u128,
            (words[2] as u128) << 64 | words[3] as u128,
        )
    }
}

#[pymethods]
impl SeedSequence {
    /// `None` draws 128 bits of OS entropy (Python's `secrets.randbits`, as numpy does). A
    /// `spawn_key` of `None` is `()` here, where numpy rejects it.
    #[new]
    #[pyo3(signature = (entropy=None, *, spawn_key=None))]
    fn new(py: Python<'_>, entropy: Option<Bound<'_, PyAny>>, spawn_key: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let numpy = numpy_types(py)?;
        let entropy = match entropy {
            None => py.import("secrets")?.getattr("randbits")?.call1((32 * POOL_SIZE,))?,
            Some(entropy) => {
                let accepted = entropy.is_instance_of::<pyo3::types::PyInt>()
                    || entropy.is_instance_of::<PyList>()
                    || entropy.is_instance_of::<PyTuple>()
                    || entropy.is_instance_of::<pyo3::types::PyRange>()
                    || is_instance(&entropy, numpy.as_ref().map(|np| &np.integer))?
                    || is_instance(&entropy, numpy.as_ref().map(|np| &np.ndarray))?;
                if !accepted {
                    return Err(PyTypeError::new_err(format!(
                        "SeedSequence expects int or sequence of ints for entropy not {}",
                        entropy.str()?
                    )));
                }
                entropy
            }
        };
        let spawn_key = match spawn_key {
            None => PyTuple::empty(py),
            Some(key) => py
                .import("builtins")?
                .getattr("tuple")?
                .call1((key,))?
                .cast_into::<PyTuple>()?,
        };
        let entropy_words = coerce_words(&entropy, numpy.as_ref())?;
        SeedSequence::from_parts(entropy.unbind(), entropy_words, spawn_key, numpy.as_ref())
    }

    #[getter]
    fn entropy(&self, py: Python<'_>) -> Py<PyAny> {
        self.entropy.clone_ref(py)
    }

    #[getter]
    fn spawn_key(&self, py: Python<'_>) -> Py<PyTuple> {
        self.spawn_key.clone_ref(py)
    }

    #[getter]
    fn pool_size(&self) -> usize {
        POOL_SIZE
    }

    #[getter]
    fn n_children_spawned(&self) -> usize {
        self.n_children_spawned.load(Ordering::Relaxed)
    }

    /// The four mixed 32-bit words.
    #[getter]
    fn pool(&self) -> Vec<u32> {
        self.pool.to_vec()
    }

    /// `generate_state(n_words, dtype)` as a list: `dtype` is `uint32` or `uint64`, as a name or
    /// numpy's type or dtype. A uint64 word is two uint32 words, low first.
    #[pyo3(signature = (n_words, dtype=None))]
    fn generate_state(&self, n_words: usize, dtype: Option<Bound<'_, PyAny>>) -> PyResult<Vec<u64>> {
        let name: String = match dtype {
            None => "uint32".to_string(),
            Some(dtype) if dtype.is_instance_of::<PyString>() => dtype.extract()?,
            Some(dtype) if dtype.hasattr("__name__")? => dtype.getattr("__name__")?.extract()?,
            Some(dtype) => dtype.getattr("name")?.extract()?,
        };
        match name.as_str() {
            "uint32" => Ok(generate_words(&self.pool, n_words).into_iter().map(u64::from).collect()),
            "uint64" => Ok(words_to_u64(&generate_words(&self.pool, 2 * n_words))),
            _ => Err(PyValueError::new_err("only support uint32 or uint64")),
        }
    }

    /// Children with spawn keys `spawn_key + (i,)`, `i` counting on from earlier spawns.
    fn spawn(&self, py: Python<'_>, n_children: usize) -> PyResult<Vec<SeedSequence>> {
        let numpy = numpy_types(py)?;
        let first = self.n_children_spawned.fetch_add(n_children, Ordering::Relaxed);
        (first..first + n_children)
            .map(|i| {
                let key = self.spawn_key.bind(py).as_any().add(PyTuple::new(py, [i])?)?;
                SeedSequence::from_parts(
                    self.entropy.clone_ref(py),
                    self.entropy_words.clone(),
                    key.cast_into::<PyTuple>()?,
                    numpy.as_ref(),
                )
            })
            .collect()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mut repr = format!("SeedSequence(entropy={}", self.entropy.bind(py).repr()?);
        if !self.spawn_key.bind(py).is_empty() {
            repr += &format!(", spawn_key={}", self.spawn_key.bind(py).repr()?);
        }
        Ok(repr + ")")
    }
}

/// PCG XSL-RR 128/64 (O'Neill 2014) as numpy's `PCG64`: a 128-bit LCG stepped before each
/// output, whose two halves are xored and rotated by the state's top six bits.
#[derive(Clone)]
struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    /// `pcg64_set_seed`: `pcg_setseq_128_srandom_r(initstate, initseq)`.
    fn seeded(initstate: u128, initseq: u128) -> Self {
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

    fn next_u64(&mut self) -> u64 {
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

/// numpy raises `OverflowError` for any integer a state field can't hold, negative ones included;
/// pyo3 raises `ValueError` for a negative one.
fn out_of_bounds<T>(py: Python<'_>, extracted: PyResult<T>) -> PyResult<T> {
    extracted.map_err(|error| {
        if error.is_instance_of::<PyValueError>(py) {
            PyOverflowError::new_err(error.value(py).to_string())
        } else {
            error
        }
    })
}

/// `np.random.default_rng(seed)`'s `Generator`: `random`, `uniform` and the dropout mask, drawn
/// from its own PCG64 in C order on the calling thread. `state` reads and writes numpy's
/// `bit_generator.state` dict.
#[pyclass(module = "indrajala_math_rust", skip_from_py_object)]
pub struct Generator {
    pcg: Pcg64,
    // numpy's buffered half of a 64-bit draw for 32-bit draws, which nothing here makes; kept so
    // a state round-trips unchanged
    has_uint32: i32,
    uinteger: u32,
}

impl Generator {
    fn from_seed_sequence(seed_sequence: &SeedSequence) -> Self {
        Generator {
            pcg: seed_sequence.pcg64_seed(),
            has_uint32: 0,
            uinteger: 0,
        }
    }

    /// The dropout mask the fused `layer_dropout_forward*` draw with this generator.
    pub(crate) fn draw_bernoulli_mask(&mut self, drop_probability: f64, size: usize) -> Vec<f64> {
        bernoulli_values(&mut self.pcg, drop_probability, size)
    }
}

#[pymethods]
impl Generator {
    /// `Generator(seed)` is `default_rng(seed)` for an int, a sequence of ints, `None` (OS
    /// entropy) or a `SeedSequence`.
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(py: Python<'_>, seed: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        if let Some(seed_sequence) = seed.as_ref().and_then(|seed| seed.cast::<SeedSequence>().ok()) {
            return Ok(Generator::from_seed_sequence(seed_sequence.get()));
        }
        Ok(Generator::from_seed_sequence(&SeedSequence::new(py, seed, None)?))
    }

    /// `rng.random(shape)`: doubles in `[0, 1)`, filled in C order.
    fn random(&mut self, shape: &Bound<'_, PyAny>) -> PyResult<RustArray> {
        let shape = parse_shape(shape)?;
        let data = (0..shape.size()).map(|_| self.pcg.next_double()).collect();
        Ok(shaped(data, shape))
    }

    /// `rng.uniform(low, high, shape)`: `low + (high - low) * u`, filled in C order. A non-finite
    /// range raises numpy's `OverflowError` and a negative one its `ValueError`, unlike the legacy
    /// `uniform`, which accepts `high < low`.
    fn uniform(&mut self, low: f64, high: f64, shape: &Bound<'_, PyAny>) -> PyResult<RustArray> {
        let shape = parse_shape(shape)?;
        let range = high - low;
        if !range.is_finite() {
            return Err(PyOverflowError::new_err("high - low range exceeds valid bounds"));
        }
        if range < 0.0 {
            return Err(PyValueError::new_err("high - low < 0"));
        }
        let data = (0..shape.size())
            .map(|_| low + range * self.pcg.next_double())
            .collect();
        Ok(shaped(data, shape))
    }

    /// `(rng.random(shape) >= drop_probability).astype(float)`: 1.0 kept, 0.0 dropped.
    fn bernoulli_mask(&mut self, drop_probability: f64, shape: &Bound<'_, PyAny>) -> PyResult<RustArray> {
        let shape = parse_shape(shape)?;
        Ok(shaped(self.draw_bernoulli_mask(drop_probability, shape.size()), shape))
    }

    /// numpy's `bit_generator.state`: `{"bit_generator": "PCG64", "state": {"state": int, "inc":
    /// int}, "has_uint32": int, "uinteger": int}`.
    #[getter]
    fn get_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = PyDict::new(py);
        inner.set_item("state", self.pcg.state)?;
        inner.set_item("inc", self.pcg.inc)?;
        let state = PyDict::new(py);
        state.set_item("bit_generator", "PCG64")?;
        state.set_item("state", inner)?;
        state.set_item("has_uint32", self.has_uint32)?;
        state.set_item("uinteger", self.uinteger)?;
        Ok(state)
    }

    /// Sets numpy's state dict, checked as numpy's setter checks it; a rejected state leaves the
    /// generator unchanged.
    #[setter]
    fn set_state(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = value
            .cast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("state must be a dict"))?;
        let bit_generator = value.get_item("bit_generator")?;
        if !bit_generator.is_some_and(|name| name.eq("PCG64").unwrap_or(false)) {
            return Err(PyValueError::new_err("state must be for a PCG64 RNG"));
        }
        let py = value.py();
        let inner = value.as_any().get_item("state")?;
        let state: u128 = out_of_bounds(py, inner.get_item("state")?.extract())?;
        let inc: u128 = out_of_bounds(py, inner.get_item("inc")?.extract())?;
        let has_uint32: i32 = out_of_bounds(py, value.as_any().get_item("has_uint32")?.extract())?;
        let uinteger: u32 = out_of_bounds(py, value.as_any().get_item("uinteger")?.extract())?;
        self.pcg = Pcg64 { state, inc };
        self.has_uint32 = has_uint32;
        self.uinteger = uinteger;
        Ok(())
    }

    /// Pickling and `copy.deepcopy`: a new generator set to this one's state.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let py = slf.py();
        Ok((
            py.get_type::<Generator>().into_any(),
            PyTuple::empty(py),
            slf.borrow().get_state(py)?,
        ))
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        self.set_state(state)
    }

    fn __repr__(&self) -> &'static str {
        "Generator(PCG64)"
    }
}

/// `np.random.default_rng(seed=None)`: a `Generator` is returned as it is; anything else seeds a
/// new one as `Generator(seed)` does.
#[pyfunction]
#[pyo3(signature = (seed=None))]
pub fn default_rng<'py>(py: Python<'py>, seed: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    if let Some(generator) = seed.as_ref().filter(|seed| seed.is_instance_of::<Generator>()) {
        return Ok(generator.clone());
    }
    Ok(Bound::new(py, Generator::new(py, seed)?)?.into_any())
}
