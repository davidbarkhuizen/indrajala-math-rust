//! The core's census (`indrajala_math::census`, the `census` feature) for Python: each process writes
//! its records at exit, to `census-<pid>.json` in `$INDRAJALA_CENSUS_DIR`, so a benchmark's worker
//! processes are counted too. indrajala-ml's `scripts/measurement/shape_census.py` adds them up.

use pyo3::prelude::*;

use indrajala_math::census;

/// Writes this process's records, when `$INDRAJALA_CENSUS_DIR` names a directory.
#[pyfunction]
fn census_write() -> PyResult<()> {
    let Ok(directory) = std::env::var("INDRAJALA_CENSUS_DIR") else {
        return Ok(());
    };
    let rows: Vec<String> = census::report()
        .iter()
        .map(|(kernel, m, k, n, calls, nanos)| format!("[\"{kernel}\",{m},{k},{n},{calls},{nanos}]"))
        .collect();
    let path = format!("{directory}/census-{}.json", std::process::id());
    std::fs::write(path, format!("[{}]\n", rows.join(",\n")))?;
    Ok(())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(census_write, m)?)?;
    m.py()
        .import("atexit")?
        .call_method1("register", (m.getattr("census_write")?,))?;
    Ok(())
}
