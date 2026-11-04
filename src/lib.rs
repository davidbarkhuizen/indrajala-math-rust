use pyo3::prelude::*;

// The modules follow the core's tree (indrajala-ml's crate structure workplan, stage 3); each
// registers its own functions and classes, under their flat Python names (D3).
mod array;
#[cfg(feature = "census")]
mod census;
mod data;
mod error;
mod linalg;
mod nn;
mod num;
mod ops;
mod program;
mod random;

/// Proves the PyO3/maturin toolchain works end to end - importable and callable from Python,
/// nothing array-specific.
#[pyfunction]
fn ping() -> PyResult<String> {
    Ok("pong".to_string())
}

// gil_used: pyo3 >= 0.28 declares a module free-threading safe unless told otherwise. This one
// hasn't been audited for free-threaded Python (Array's &mut self methods, the shared RNG
// state), so it keeps the GIL on 3.13t/3.14t.
#[pymodule(gil_used = true)]
fn indrajala_math_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    array::register(m)?;
    linalg::register(m)?;
    ops::register(m)?;
    random::register(m)?;
    nn::register(m)?;
    program::register(m)?;
    data::register(m)?;
    #[cfg(feature = "census")]
    census::register(m)?;
    random::legacy::seed_at_import();
    Ok(())
}
