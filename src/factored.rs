//! The bindings of the core's factored recurrence (`indrajala_math::factored`): list arguments
//! lent to the core as plain references, and `RecurrenceStep`, which keeps the tokens a step read
//! for its backward pass.

use pyo3::prelude::*;

use crate::array::RustArray;
use crate::linalg::py_err;
#[allow(unused_imports)]
pub(crate) use indrajala_math::factored::*;
use indrajala_math::Array;

/// Each row of `h`, `(N, S)`, snapped factor by factor: factor `g`'s `sizes[g]` columns to the
/// nearest of its codes `codes[g]` (`codes_forward`), or kept for a continuous factor (None).
/// Returns `(q, indices)`, `q` `(N, S)` and each factor's indices, None for a continuous factor.
#[pyfunction]
pub fn factored_codes_forward(
    h: &RustArray,
    sizes: Vec<usize>,
    codes: Vec<Option<PyRef<'_, RustArray>>>,
) -> PyResult<(RustArray, Vec<Option<Vec<usize>>>)> {
    indrajala_math::factored::factored_codes_forward(h, sizes, &option_refs(&codes))
        .map(|(q, indices)| (q.into(), indices))
        .map_err(py_err)
}

/// `factored_codes_forward`'s backward pass: factor by factor, `codes_backward` of its columns of
/// `dq`, `h` and `q` with its own commitment and its commitment target `targets[g]` (None: its
/// `q`; no `targets`, every factor's `q`), or `dq`'s columns straight through for a continuous
/// factor (its `grad_codes` None). Returns `(dh, grad_codes)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (dq, h, q, sizes, indices, grad_codes, weight, commitments, targets = None))]
pub fn factored_codes_backward(
    dq: &RustArray,
    h: &RustArray,
    q: &RustArray,
    sizes: Vec<usize>,
    indices: Vec<Option<Vec<usize>>>,
    grad_codes: Vec<Option<PyRef<'_, RustArray>>>,
    weight: f64,
    commitments: Vec<f64>,
    targets: Option<Vec<Option<PyRef<'_, RustArray>>>>,
) -> PyResult<(RustArray, Vec<Option<RustArray>>)> {
    let targets = targets.as_deref().map(option_refs);
    indrajala_math::factored::factored_codes_backward(
        dq,
        h,
        q,
        sizes,
        indices,
        &option_refs(&grad_codes),
        weight,
        commitments,
        targets.as_deref(),
    )
    .map(|(dh, grad_codes)| {
        (
            dh.into(),
            grad_codes.into_iter().map(|e| e.map(RustArray::from)).collect(),
        )
    })
    .map_err(py_err)
}

/// A window's factored recurrence step, from `recurrence_step_forward`: the next state, the cells'
/// outputs before they are snapped, and each factor's code indices, with what
/// `recurrence_step_backward` reads.
#[pyclass(module = "indrajala_math_rust", frozen, skip_from_py_object)]
pub struct RecurrenceStep {
    /// the tokens the poolings read: one array for every factor, or one a factor
    x: StepTokens,
    /// the step, its state and cells' outputs moved to `state` and `h`
    core: StepCore,
    state: Py<RustArray>,
    h: Py<RustArray>,
}

#[pymethods]
impl RecurrenceStep {
    /// The next state, `(N, S)`: each factor's code, or its cell's output if continuous.
    #[getter]
    fn state(&self, py: Python<'_>) -> Py<RustArray> {
        self.state.clone_ref(py)
    }

    /// The cells' outputs before they are snapped, `(N, S)`.
    #[getter]
    fn h(&self, py: Python<'_>) -> Py<RustArray> {
        self.h.clone_ref(py)
    }

    /// Each factor's code indices, one per example; None for a continuous factor.
    #[getter]
    fn indices(&self) -> Vec<Option<Vec<usize>>> {
        self.core.indices()
    }

    /// The factors' state sizes.
    #[getter]
    fn sizes(&self) -> Vec<usize> {
        self.core.sizes.clone()
    }
}

/// The arrays a list argument holds, as plain references.
fn refs<'a>(arrays: &'a [PyRef<'_, RustArray>]) -> Vec<&'a Array> {
    arrays.iter().map(|a| &a.0).collect()
}

/// A `targets` argument as plain references, each factor's (None: its `q`); no `targets`, None for
/// each of `count` factors.
fn target_refs<'a>(targets: Option<&'a [Option<PyRef<'_, RustArray>>]>, count: usize) -> Vec<Option<&'a Array>> {
    match targets {
        Some(targets) => targets.iter().map(|t| t.as_deref().map(|a| &a.0)).collect(),
        None => vec![None; count],
    }
}

/// A list argument of optional arrays as plain references.
fn option_refs<'a>(arrays: &'a [Option<PyRef<'_, RustArray>>]) -> Vec<Option<&'a Array>> {
    arrays.iter().map(|a| a.as_deref().map(|a| &a.0)).collect()
}

/// The tokens a step's poolings read, as `recurrence_step_forward` was given them.
enum StepTokens {
    /// one array every factor's pooling reads
    Shared(Py<RustArray>),
    /// factor `g`'s pooling reads the `g`-th (the window program's branches)
    Each(Vec<Py<RustArray>>),
}

/// A window's recurrence over G factors in one call: from the tokens the window's output layer
/// reads, `x` `(N, T * d)` (or a list of G such arrays, factor `g`'s pooling reading the `g`-th),
/// and the state `c` `(N, S)`, each factor `g`'s pooling
/// (`queries[g]`, `(d,)`), cell (`w[g]`, `u_weights[g]`, `b[g]`, as `gru_forward`'s, `w[g]` `(3
/// s_g, d)`, or `(3 s_g, d + S - s_g)` with `cross`) and codes (`codes[g]`, `(m_g, s_g)`, or None
/// for a continuous factor), in factor order; with `temperatures`, factor `g` soft at
/// `temperatures[g]` (`soft_codes_forward`; None, or no `temperatures`, hard); with `couplings`,
/// factor `g`'s next state `(1 - couplings[g]) h_g + couplings[g] q_g` (no `couplings`, every 1: its
/// `q_g`). Returns the `RecurrenceStep`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (x, c, queries, w, u_weights, b, codes, tokens, *, cross, temperatures = None, couplings = None))]
pub fn recurrence_step_forward(
    py: Python<'_>,
    x: Bound<'_, PyAny>,
    c: &RustArray,
    queries: Vec<PyRef<'_, RustArray>>,
    w: Vec<PyRef<'_, RustArray>>,
    u_weights: Vec<PyRef<'_, RustArray>>,
    b: Vec<PyRef<'_, RustArray>>,
    codes: Vec<Option<PyRef<'_, RustArray>>>,
    tokens: usize,
    cross: bool,
    temperatures: Option<Vec<Option<f64>>>,
    couplings: Option<Vec<f64>>,
) -> PyResult<RecurrenceStep> {
    let codes = option_refs(&codes);
    let temperatures = temperatures.unwrap_or_else(|| vec![None; codes.len()]);
    let couplings = couplings.unwrap_or_else(|| vec![1.0; codes.len()]);
    let x = match x.cast::<RustArray>() {
        Ok(one) => StepTokens::Shared(one.clone().unbind()),
        Err(_) => StepTokens::Each(x.extract::<Vec<Py<RustArray>>>()?),
    };
    let borrowed: Vec<PyRef<'_, RustArray>> = match &x {
        StepTokens::Shared(one) => (0..queries.len()).map(|_| one.bind(py).borrow()).collect(),
        StepTokens::Each(each) => each.iter().map(|a| a.bind(py).borrow()).collect(),
    };
    let core = step_forward(
        &refs(&borrowed),
        c,
        &refs(&queries),
        &refs(&w),
        &refs(&u_weights),
        &refs(&b),
        &codes,
        &temperatures,
        &couplings,
        tokens,
        cross,
        "recurrence_step_forward",
    )
    .map_err(py_err)?;
    drop(borrowed);
    let mut core = core;
    let state = std::mem::replace(&mut core.state, Array::from_vector(Vec::new()));
    let h = std::mem::replace(&mut core.h, Array::from_vector(Vec::new()));
    Ok(RecurrenceStep {
        state: Py::new(py, RustArray::from(state))?,
        h: Py::new(py, RustArray::from(h))?,
        x,
        core,
    })
}

/// `recurrence_step_forward`'s backward pass, from the next state's gradient `dq` `(N, S)`: factor
/// by factor, its codes' (`codes_backward`, weight, its `commitments[g]` and its commitment target
/// `targets[g]`, None or no `targets` its `q`; straight through if continuous), its cell's (`gru_backward`) and its pooling's (`attention_pooling_backward`), each
/// parameter's gradient accumulated onto the one given. Returns `(dx, dc, grad_queries, grad_w,
/// grad_u, grad_b, grad_codes)`: the tokens' gradient (one array, the factors' poolings' summed in
/// factor order, if the step read one; a list, each factor's, if it read one a factor), the
/// state's, and each factor's parameters'. With `straight`, a soft factor's gradient is straight
/// through (`soft_codes_backward`'s).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (step, dq, queries, w, u_weights, grad_queries, grad_w, grad_u, grad_b, grad_codes, weight, commitments, targets = None, straight = false))]
pub fn recurrence_step_backward(
    py: Python<'_>,
    step: PyRef<'_, RecurrenceStep>,
    dq: &RustArray,
    queries: Vec<PyRef<'_, RustArray>>,
    w: Vec<PyRef<'_, RustArray>>,
    u_weights: Vec<PyRef<'_, RustArray>>,
    grad_queries: Vec<PyRef<'_, RustArray>>,
    grad_w: Vec<PyRef<'_, RustArray>>,
    grad_u: Vec<PyRef<'_, RustArray>>,
    grad_b: Vec<PyRef<'_, RustArray>>,
    grad_codes: Vec<Option<PyRef<'_, RustArray>>>,
    weight: f64,
    commitments: Vec<f64>,
    targets: Option<Vec<Option<PyRef<'_, RustArray>>>>,
    straight: bool,
) -> PyResult<(
    Py<PyAny>,
    RustArray,
    Vec<RustArray>,
    Vec<RustArray>,
    Vec<RustArray>,
    Vec<RustArray>,
    Vec<Option<RustArray>>,
)> {
    let borrowed: Vec<PyRef<'_, RustArray>> = match &step.x {
        StepTokens::Shared(one) => (0..queries.len()).map(|_| one.bind(py).borrow()).collect(),
        StepTokens::Each(each) => each.iter().map(|a| a.bind(py).borrow()).collect(),
    };
    let codes = option_refs(&grad_codes);
    let (dxs, dc, gq, gw, gu, gb, ge) = step
        .core
        .backward(
            &refs(&borrowed),
            dq,
            &refs(&queries),
            &refs(&w),
            &refs(&u_weights),
            &StepGradients {
                queries: &refs(&grad_queries),
                w: &refs(&grad_w),
                u: &refs(&grad_u),
                b: &refs(&grad_b),
                codes: &codes,
            },
            weight,
            &commitments,
            &target_refs(targets.as_deref(), commitments.len()),
            straight,
            "recurrence_step_backward",
        )
        .map_err(py_err)?;
    let dx = match &step.x {
        StepTokens::Shared(_) => {
            let all: Vec<usize> = (0..dxs.len()).collect();
            Py::new(
                py,
                RustArray::from(sum_in_order(&dxs, &all).expect("a step has 1 or more factors")),
            )?
            .into_any()
        }
        StepTokens::Each(_) => dxs
            .into_iter()
            .map(|dx| Py::new(py, RustArray::from(dx)))
            .collect::<PyResult<Vec<_>>>()?
            .into_pyobject(py)?
            .into_any()
            .unbind(),
    };
    let arrays = |v: Vec<Array>| v.into_iter().map(RustArray::from).collect::<Vec<_>>();
    Ok((
        dx,
        dc.into(),
        arrays(gq),
        arrays(gw),
        arrays(gu),
        arrays(gb),
        ge.into_iter().map(|e| e.map(RustArray::from)).collect(),
    ))
}
