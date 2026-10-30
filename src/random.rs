//! The bindings of the core's global generator (`indrajala_math::random`): the seed's Python
//! forms read as numpy's `np.random.seed` reads them, and the draws in a shape from Python.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::array::{parse_shape, RustArray};
use crate::linalg::py_err;
use indrajala_math::random as core_random;
#[allow(unused_imports)]
pub(crate) use indrajala_math::random::{bernoulli_values, draw_bernoulli_mask, seed_at_import, shaped, DoubleSource};

const SEED_RANGE_MESSAGE: &str = "Seed must be between 0 and 2**32 - 1";

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

    fn sequence(&mut self, depth: usize, items: Vec<Bound<'_, PyAny>>, safe_kind: Option<bool>) -> PyResult<()> {
        if self.ndim.is_some_and(|ndim| depth >= ndim) {
            return Err(inhomogeneous(depth));
        }
        match self.dims.get(depth) {
            Some(&len) if len != items.len() => return Err(inhomogeneous(depth)),
            Some(_) => {}
            None => self.dims.push(items.len()),
        }
        for item in items {
            self.discover(&item, depth + 1, safe_kind)?;
        }
        Ok(())
    }

    /// Walks a seed the way `np.asarray` discovers its shape and dtype. `safe_kind` is set inside
    /// a buffer, whose format fixes the dtype of every element (`Some(false)`: none survive).
    fn discover(&mut self, obj: &Bound<'_, PyAny>, depth: usize, safe_kind: Option<bool>) -> PyResult<()> {
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
                self.sequence(depth, obj.try_iter()?.collect::<PyResult<_>>()?, Some(safe))
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
            return self.discover(&as_list, depth, Some(safe));
        }
        if unsafe { pyo3::ffi::PySequence_Check(obj.as_ptr()) } == 1 {
            return self.sequence(depth, obj.try_iter()?.collect::<PyResult<_>>()?, safe_kind);
        }
        // a Python int within int64 is cast safely; anything else numpy holds as a float,
        // complex or object dtype, and none of those cast
        let word = if safe_kind != Some(false) && obj.is_instance_of::<pyo3::types::PyInt>() {
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
fn key_from_sequence(seed: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
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
pub fn seed(py: Python<'_>, seed: Option<Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(mut seed) = seed else {
        core_random::reseed_from_entropy();
        return Ok(());
    };
    if seed.hasattr("squeeze")? {
        seed = seed.call_method0("squeeze")?;
    }
    match py.import("operator")?.getattr("index")?.call1((&seed,)) {
        Ok(index) => {
            let word: u32 = index.extract().map_err(|_| PyValueError::new_err(SEED_RANGE_MESSAGE))?;
            core_random::seed_word(word);
        }
        Err(error) if error.is_instance_of::<PyTypeError>(py) => core_random::seed_key(&key_from_sequence(&seed)?),
        Err(error) => return Err(error),
    }
    Ok(())
}

/// `np.random.random(shape)`: uniform draws in `[0, 1)`, filled in C order.
#[pyfunction]
pub fn random(shape: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    Ok(core_random::random(parse_shape(shape)?).into())
}

/// `np.random.uniform(low, high, size=shape)`, `randomize()`'s draw: `low + range * u` with
/// `range = high - low` computed once, filled in C order. A non-finite range raises numpy's
/// `OverflowError`.
#[pyfunction]
pub fn uniform(low: f64, high: f64, shape: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    core_random::uniform(low, high, parse_shape(shape)?)
        .map(RustArray::from)
        .map_err(py_err)
}

/// `(np.random.random(shape) >= drop_probability).astype(float)`: `draw_bernoulli_mask` as an
/// array, so the mask can be checked against numpy on its own.
#[pyfunction]
pub fn bernoulli_mask(drop_probability: f64, shape: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    Ok(core_random::bernoulli_mask(drop_probability, parse_shape(shape)?).into())
}
