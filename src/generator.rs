//! The bindings of the core's `default_rng` (`indrajala_math::generator`): a seed's Python forms
//! read as numpy reads them, and `bit_generator.state` as numpy's dict.

use std::sync::atomic::{AtomicUsize, Ordering};

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::array::{parse_shape, RustArray};
use crate::linalg::py_err;
use indrajala_math::generator::{GeneratorState, POOL_SIZE};

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
    inner: indrajala_math::generator::SeedSequence,
    n_children_spawned: AtomicUsize,
}

impl SeedSequence {
    fn from_parts(
        entropy: Py<PyAny>,
        entropy_words: Vec<u32>,
        spawn_key: Bound<'_, PyTuple>,
        numpy: Option<&NumpyTypes<'_>>,
    ) -> PyResult<Self> {
        let spawn_words = coerce_words(spawn_key.as_any(), numpy)?;
        Ok(SeedSequence {
            inner: indrajala_math::generator::SeedSequence::from_words(&entropy_words, &spawn_words),
            entropy,
            spawn_key: spawn_key.unbind(),
            entropy_words,
            n_children_spawned: AtomicUsize::new(0),
        })
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
        self.inner.pool().to_vec()
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
            "uint32" => Ok(self
                .inner
                .generate_state_u32(n_words)
                .into_iter()
                .map(u64::from)
                .collect()),
            "uint64" => Ok(self.inner.generate_state_u64(n_words)),
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
pub struct Generator(pub indrajala_math::generator::Generator);

impl std::ops::Deref for Generator {
    type Target = indrajala_math::generator::Generator;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Generator {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
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
            return Ok(Generator(indrajala_math::generator::Generator::from_seed_sequence(
                &seed_sequence.get().inner,
            )));
        }
        Ok(Generator(indrajala_math::generator::Generator::from_seed_sequence(
            &SeedSequence::new(py, seed, None)?.inner,
        )))
    }

    /// `rng.random(shape)`: doubles in `[0, 1)`, filled in C order.
    fn random(&mut self, shape: &Bound<'_, PyAny>) -> PyResult<RustArray> {
        Ok(self.0.random(parse_shape(shape)?).into())
    }

    /// `rng.uniform(low, high, shape)`: `low + (high - low) * u`, filled in C order. A non-finite
    /// range raises numpy's `OverflowError` and a negative one its `ValueError`, unlike the legacy
    /// `uniform`, which accepts `high < low`.
    fn uniform(&mut self, low: f64, high: f64, shape: &Bound<'_, PyAny>) -> PyResult<RustArray> {
        self.0
            .uniform(low, high, parse_shape(shape)?)
            .map(RustArray::from)
            .map_err(py_err)
    }

    /// `(rng.random(shape) >= drop_probability).astype(float)`: 1.0 kept, 0.0 dropped.
    fn bernoulli_mask(&mut self, drop_probability: f64, shape: &Bound<'_, PyAny>) -> PyResult<RustArray> {
        Ok(self.0.bernoulli_mask(drop_probability, parse_shape(shape)?).into())
    }

    /// numpy's `bit_generator.state`: `{"bit_generator": "PCG64", "state": {"state": int, "inc":
    /// int}, "has_uint32": int, "uinteger": int}`.
    #[getter]
    fn get_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let current = self.0.state();
        let inner = PyDict::new(py);
        inner.set_item("state", current.state)?;
        inner.set_item("inc", current.inc)?;
        let state = PyDict::new(py);
        state.set_item("bit_generator", "PCG64")?;
        state.set_item("state", inner)?;
        state.set_item("has_uint32", current.has_uint32)?;
        state.set_item("uinteger", current.uinteger)?;
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
        self.0.set_state(GeneratorState {
            state,
            inc,
            has_uint32,
            uinteger,
        });
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
