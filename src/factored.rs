//! A factored window-recurrent state (indrajala-ml's factored-state workplan, stage 2): G factors,
//! each its own pooling, GRU cell and codes, the state `(N, S)` their states side by side, factor
//! `g` the `s_g` columns after the factors before it (`S = s_1 + ... + s_G`).
//!
//! ```text
//! pooling g     u_g = pool(x; q_g)                       (attention_pooling_forward)
//! cell g        h_g = GRU_g(c_g, i_g),  i_g = u_g, or [u_g, c_(-g)] with the cross-term on
//! codes g       q_g = e_(g, k), k = argmin ||h_g - e_g,k||^2   (codes_forward); q_g = h_g if continuous
//! next state    [q_1, ..., q_G]
//! ```
//!
//! `c_(-g)` is the other factors' states in factor order, the state's columns without factor
//! `g`'s; factor `g`'s `w` is then `(3 s_g, d + S - s_g)`, its input `d` first. Each step is the
//! single-factor op it names in `recurrent.rs`, called factor by factor in factor order, so with
//! G = 1 the forward and backward passes are today's by bits (the workplan's D6), and every
//! factor's are those ops' at any G. The sums the factors share are left folds in factor order,
//! each starting from factor 1's term (not from `0.0`, which would turn a `-0.0` into `0.0`):
//! the tokens' gradient `dx = dx_1 + ... + dx_G` (every factor's pooling reads the same tokens),
//! and, with the cross-term, factor `g`'s state gradient its own cell's, then each other factor's
//! input gradient through its `c_(-h)` columns, in factor order.
//!
//! `recurrence_step_forward` runs a window's every factor in one call and gives a
//! `RecurrenceStep`, which holds what `recurrence_step_backward` reads; `factored_codes_*` snap a
//! state over the factors' codes alone (the learned initial state, W8).

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{RustArray, Shape};
use crate::ops::same_shape_elementwise;
use crate::recurrent::{
    attention_pooling_backward, attention_pooling_forward, codes_backward, codes_forward, columns, features_per_token,
    gru_backward, gru_forward, matrix_shape, require_shape, require_temperature, soft_codes_backward,
    soft_codes_forward,
};

/// Each factor's first column, from the factors' sizes, every one 1 or more; and their total.
fn factor_starts(sizes: &[usize], context: &str) -> PyResult<(Vec<usize>, usize)> {
    if sizes.is_empty() || sizes.contains(&0) {
        return Err(PyValueError::new_err(format!(
            "{context} requires 1 or more factors, each of size 1 or more, got sizes {sizes:?}"
        )));
    }
    let mut starts = Vec::with_capacity(sizes.len());
    let mut total = 0;
    for &size in sizes {
        starts.push(total);
        total += size;
    }
    Ok((starts, total))
}

/// The factors' blocks of `x`, `(rows, total)`: each its own columns, a copy.
fn split(x: &RustArray, total: usize, starts: &[usize], sizes: &[usize]) -> Vec<RustArray> {
    starts
        .iter()
        .zip(sizes)
        .map(|(&start, &size)| columns(x, total, start, start + size))
        .collect()
}

/// The factors' blocks side by side, `(rows, total)`: `split`'s inverse.
fn join(blocks: &[RustArray], rows: usize, total: usize) -> RustArray {
    let mut data = Vec::with_capacity(rows * total);
    for row in 0..rows {
        for block in blocks {
            let width = block.data.len() / rows.max(1);
            data.extend_from_slice(&block.data[row * width..(row + 1) * width]);
        }
    }
    RustArray::from_matrix(data, rows, total)
}

/// The list arguments' common length, the factor count G, refused unless they agree.
fn factor_count(lengths: &[(usize, &str)], context: &str) -> PyResult<usize> {
    let (count, _) = lengths[0];
    if let Some((length, name)) = lengths.iter().find(|(length, _)| *length != count) {
        return Err(PyValueError::new_err(format!(
            "{context} requires one entry per factor in every list, {count} factors, got {length} in {name}"
        )));
    }
    Ok(count)
}

/// A soft factor's forward pass, what its backward pass reads (indrajala-ml's commitment-control
/// workplan, stage 3).
struct Soft {
    temperature: f64,
    /// the weights, `(N, m_g)`
    weights: RustArray,
    /// the codes the step read, a copy, `(m_g, s_g)`
    codes: RustArray,
}

/// Factor `g`'s state `c_g` snapped to its nearest code, or with a temperature its soft state
/// (`soft_codes_forward`), or itself for a continuous factor: `(q, indices, soft)`, the indices
/// the nearest codes either way.
fn snap(
    h: &RustArray,
    codes: Option<&RustArray>,
    temperature: Option<f64>,
) -> PyResult<(RustArray, Option<Vec<usize>>, Option<Soft>)> {
    match (codes, temperature) {
        (Some(e), None) => {
            let (q, indices) = codes_forward(h, e)?;
            Ok((q, Some(indices), None))
        }
        (Some(e), Some(temperature)) => {
            let (q, weights, indices) = soft_codes_forward(h, e, temperature)?;
            let soft = Soft {
                temperature,
                weights,
                codes: e.clone(),
            };
            Ok((q, Some(indices), Some(soft)))
        }
        (None, None) => Ok((h.clone(), None, None)),
        (None, Some(_)) => Err(PyValueError::new_err(
            "a continuous factor has no codes to soften: its temperature must be None",
        )),
    }
}

/// `snap`'s backward pass: `codes_backward` (with the commitment's `target`, if any), or
/// `soft_codes_backward` for a soft factor (no target), or `dq` itself for a continuous factor.
/// Returns `(dh, grad_e)`.
#[allow(clippy::too_many_arguments)]
fn snap_backward(
    dq: RustArray,
    h: &RustArray,
    q: &RustArray,
    indices: &Option<Vec<usize>>,
    soft: Option<&Soft>,
    grad_e: Option<&RustArray>,
    weight: f64,
    commitment: f64,
    target: Option<&RustArray>,
    context: &str,
) -> PyResult<(RustArray, Option<RustArray>)> {
    if let (Some(soft), Some(grad_e)) = (soft, grad_e) {
        if target.is_some() {
            return Err(PyValueError::new_err(format!(
                "{context} requires None as a soft factor's commitment target: a consolidation needs hard codes"
            )));
        }
        let (dh, grad_e) = soft_codes_backward(
            &dq,
            h,
            q,
            &soft.weights,
            &soft.codes,
            grad_e,
            weight,
            commitment,
            soft.temperature,
        )?;
        return Ok((dh, Some(grad_e)));
    }
    match (indices, grad_e) {
        (Some(indices), Some(grad_e)) => {
            let (dh, grad_e) = codes_backward(&dq, h, q, indices.clone(), grad_e, weight, commitment, target)?;
            Ok((dh, Some(grad_e)))
        }
        (None, None) => match target {
            None => Ok((dq, None)),
            Some(_) => Err(PyValueError::new_err(format!(
                "{context} requires None as a continuous factor's commitment target"
            ))),
        },
        (Some(_), None) => Err(PyValueError::new_err(format!(
            "{context} requires a codes gradient for each snapped factor, got None"
        ))),
        (None, Some(_)) => Err(PyValueError::new_err(format!(
            "{context} requires None as a continuous factor's codes gradient"
        ))),
    }
}

/// Each row of `h`, `(N, S)`, snapped factor by factor: factor `g`'s `sizes[g]` columns to the
/// nearest of its codes `codes[g]` (`codes_forward`), or kept for a continuous factor (None).
/// Returns `(q, indices)`, `q` `(N, S)` and each factor's indices, None for a continuous factor.
#[pyfunction]
pub fn factored_codes_forward(
    h: &RustArray,
    sizes: Vec<usize>,
    codes: Vec<Option<PyRef<'_, RustArray>>>,
) -> PyResult<(RustArray, Vec<Option<Vec<usize>>>)> {
    let context = "factored_codes_forward";
    factor_count(&[(sizes.len(), "sizes"), (codes.len(), "codes")], context)?;
    let (starts, total) = factor_starts(&sizes, context)?;
    let (rows, _) = matrix_shape(h, "h", context)?;
    require_shape(h, Shape::Matrix(rows, total), "h", context)?;
    let mut blocks = Vec::with_capacity(sizes.len());
    let mut indices = Vec::with_capacity(sizes.len());
    for (h_g, e) in split(h, total, &starts, &sizes).iter().zip(&codes) {
        let (q, i, _) = snap(h_g, e.as_deref(), None)?;
        blocks.push(q);
        indices.push(i);
    }
    Ok((join(&blocks, rows, total), indices))
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
    let context = "factored_codes_backward";
    let targets = target_refs(targets.as_deref(), sizes.len());
    factor_count(
        &[
            (sizes.len(), "sizes"),
            (indices.len(), "indices"),
            (grad_codes.len(), "grad_codes"),
            (commitments.len(), "commitments"),
            (targets.len(), "targets"),
        ],
        context,
    )?;
    let (starts, total) = factor_starts(&sizes, context)?;
    let (rows, _) = matrix_shape(h, "h", context)?;
    for (x, name) in [(dq, "dq"), (h, "h"), (q, "q")] {
        require_shape(x, Shape::Matrix(rows, total), name, context)?;
    }
    let mut dh = Vec::with_capacity(sizes.len());
    let mut new_grad_codes = Vec::with_capacity(sizes.len());
    let blocks = split(dq, total, &starts, &sizes)
        .into_iter()
        .zip(split(h, total, &starts, &sizes))
        .zip(split(q, total, &starts, &sizes));
    for (g, ((dq_g, h_g), q_g)) in blocks.enumerate() {
        let (dh_g, grad_e) = snap_backward(
            dq_g,
            &h_g,
            &q_g,
            &indices[g],
            None,
            grad_codes[g].as_deref(),
            weight,
            commitments[g],
            targets[g],
            context,
        )?;
        dh.push(dh_g);
        new_grad_codes.push(grad_e);
    }
    Ok((join(&dh, rows, total), new_grad_codes))
}

/// One factor's forward pass, what its backward pass reads.
struct FactorTape {
    /// the pooling's weights, `(N, T)`
    p: RustArray,
    /// the cell's input `i_g`, `(N, d)` or `(N, d + S - s_g)`
    input: RustArray,
    /// the factor's state `c_g`, `(N, s_g)`
    c: RustArray,
    r: RustArray,
    z: RustArray,
    h_tilde: RustArray,
    rc: RustArray,
    /// the cell's output before it is snapped, `(N, s_g)`
    h: RustArray,
    /// the next state's block, `(N, s_g)`
    q: RustArray,
    indices: Option<Vec<usize>>,
    /// a soft factor's weights and codes, None if hard or continuous
    soft: Option<Soft>,
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

/// Factor `g`'s state size `s_g` from its `u_weights`, `(3 s_g, s_g)`.
fn cell_size(u_weights: &RustArray, context: &str) -> PyResult<usize> {
    let (rows, size) = matrix_shape(u_weights, "every u_weights", context)?;
    if rows != 3 * size || size == 0 {
        return Err(PyValueError::new_err(format!(
            "{context} requires each u_weights of shape (3s, s), s >= 1, got {:?}",
            u_weights.shape
        )));
    }
    Ok(size)
}

/// The columns of `c` that aren't factor `g`'s, in factor order, `(N, S - s_g)`.
fn others(c: &RustArray, total: usize, start: usize, size: usize) -> RustArray {
    let rows = c.data.len() / total;
    let mut data = Vec::with_capacity(rows * (total - size));
    for row in c.data.chunks_exact(total) {
        data.extend_from_slice(&row[..start]);
        data.extend_from_slice(&row[start + size..]);
    }
    RustArray::from_matrix(data, rows, total - size)
}

/// A window's recurrence step, as `recurrence_step_forward` computes it, without Python's
/// references: what `RecurrenceStep` holds, and the window program's step (program.rs).
pub(crate) struct StepCore {
    tokens: usize,
    cross: bool,
    pub(crate) features: usize,
    pub(crate) rows: usize,
    pub(crate) sizes: Vec<usize>,
    pub(crate) starts: Vec<usize>,
    factors: Vec<FactorTape>,
    /// the next state, `(N, S)`
    pub(crate) state: RustArray,
    /// the cells' outputs before they are snapped, `(N, S)`
    pub(crate) h: RustArray,
}

impl StepCore {
    /// A step of no factors, a placeholder until `step_forward` gives one.
    pub(crate) fn empty() -> StepCore {
        StepCore {
            tokens: 0,
            cross: false,
            features: 0,
            rows: 0,
            sizes: Vec::new(),
            starts: Vec::new(),
            factors: Vec::new(),
            state: RustArray::from_vector(Vec::new()),
            h: RustArray::from_vector(Vec::new()),
        }
    }

    /// How many values the step keeps for its backward pass.
    pub(crate) fn floats(&self) -> usize {
        let tapes: usize = self
            .factors
            .iter()
            .map(|f| {
                [&f.p, &f.input, &f.c, &f.r, &f.z, &f.h_tilde, &f.rc, &f.h, &f.q]
                    .iter()
                    .map(|a| a.data.len())
                    .sum::<usize>()
                    + f.soft.as_ref().map_or(0, |s| s.weights.data.len() + s.codes.data.len())
            })
            .sum();
        tapes + self.state.data.len() + self.h.data.len()
    }

    /// Each factor's code indices, one per example; None for a continuous factor.
    pub(crate) fn indices(&self) -> Vec<Option<Vec<usize>>> {
        self.factors.iter().map(|f| f.indices.clone()).collect()
    }
}

/// The arrays a list argument holds, as plain references.
fn refs<'a>(arrays: &'a [PyRef<'_, RustArray>]) -> Vec<&'a RustArray> {
    arrays.iter().map(|a| &**a).collect()
}

/// A `targets` argument as plain references, each factor's (None: its `q`); no `targets`, None for
/// each of `count` factors.
fn target_refs<'a>(targets: Option<&'a [Option<PyRef<'_, RustArray>>]>, count: usize) -> Vec<Option<&'a RustArray>> {
    match targets {
        Some(targets) => targets.iter().map(|t| t.as_deref()).collect(),
        None => vec![None; count],
    }
}

/// Codes `e`'s `(count, size)`; a vector, which no factor's codes are, `(0, 0)`.
fn code_shape(e: &RustArray) -> (usize, usize) {
    match e.shape {
        Shape::Matrix(count, size) => (count, size),
        Shape::Vector(_) => (0, 0),
    }
}

/// The consolidation table (indrajala-ml's code-consolidation workplan, D1): each duplicate joint
/// code, a code index per factor, to its representative's, checked against the factors' `codes`
/// (every factor snapped, every index one of its factor's codes, each duplicate once).
pub(crate) struct Consolidation {
    table: HashMap<Vec<usize>, Vec<usize>>,
}

impl Consolidation {
    pub(crate) fn new(
        pairs: Vec<(Vec<usize>, Vec<usize>)>,
        codes: &[Option<&RustArray>],
        context: &str,
    ) -> PyResult<Consolidation> {
        if !pairs.is_empty() && codes.iter().any(Option::is_none) {
            return Err(PyValueError::new_err(format!(
                "{context} consolidates only when every factor is snapped: a continuous factor has no codes"
            )));
        }
        let counts: Vec<usize> = codes.iter().map(|e| e.map_or(0, |e| code_shape(e).0)).collect();
        let mut table = HashMap::with_capacity(pairs.len());
        for (duplicate, representative) in pairs {
            for code in [&duplicate, &representative] {
                if code.len() != counts.len() || code.iter().zip(&counts).any(|(&i, &m)| i >= m) {
                    return Err(PyValueError::new_err(format!(
                        "{context} requires each consolidation code as a code index per factor, below its factor's \
                         code counts {counts:?}, got {code:?}"
                    )));
                }
            }
            if let Some(earlier) = table.insert(duplicate.clone(), representative) {
                return Err(PyValueError::new_err(format!(
                    "{context} requires each duplicate once in the consolidation, got {duplicate:?} twice (first to \
                     {earlier:?})"
                )));
            }
        }
        Ok(Consolidation { table })
    }

    /// Each factor's commitment target after `step`: the rows of its codes at each example's
    /// representative's index, or its own where its joint code is no duplicate (then its `q`'s
    /// row, the same bits). None for every factor when the table is empty, so the step's backward
    /// pass is as without one.
    pub(crate) fn targets(&self, step: &StepCore, codes: &[Option<&RustArray>]) -> Vec<Option<RustArray>> {
        if self.table.is_empty() {
            return vec![None; codes.len()];
        }
        let own: Vec<&Vec<usize>> = step
            .factors
            .iter()
            .map(|f| f.indices.as_ref().expect("a consolidated factor is snapped"))
            .collect();
        let rows = own.first().map_or(0, |i| i.len());
        let mut key = vec![0; own.len()];
        let mut chosen: Vec<Vec<usize>> = vec![Vec::with_capacity(rows); own.len()];
        for row in 0..rows {
            for (k, i) in key.iter_mut().zip(&own) {
                *k = i[row];
            }
            let to = self.table.get(&key).unwrap_or(&key);
            for (c, &t) in chosen.iter_mut().zip(to) {
                c.push(t);
            }
        }
        chosen
            .iter()
            .zip(codes)
            .map(|(t, e)| {
                let e = e.expect("a consolidated factor is snapped");
                let size = code_shape(e).1;
                let mut data = Vec::with_capacity(rows * size);
                for &i in t {
                    data.extend_from_slice(&e.data[i * size..(i + 1) * size]);
                }
                Some(RustArray::from_matrix(data, rows, size))
            })
            .collect()
    }
}

/// `recurrence_step_forward`'s computation over plain references, under `context`'s name: factor
/// `g`'s pooling reads `xs[g]` (the same tokens for every factor, or each its own: the window
/// program's branches), every one of the same shape; factor `g` soft with `temperatures[g]`
/// (None: hard).
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_forward(
    xs: &[&RustArray],
    c: &RustArray,
    queries: &[&RustArray],
    w: &[&RustArray],
    u_weights: &[&RustArray],
    b: &[&RustArray],
    codes: &[Option<&RustArray>],
    temperatures: &[Option<f64>],
    tokens: usize,
    cross: bool,
    context: &str,
) -> PyResult<StepCore> {
    factor_count(
        &[
            (xs.len(), "x"),
            (queries.len(), "queries"),
            (w.len(), "w"),
            (u_weights.len(), "u_weights"),
            (b.len(), "b"),
            (codes.len(), "codes"),
            (temperatures.len(), "temperatures"),
        ],
        context,
    )?;
    for (temperature, e) in temperatures.iter().zip(codes) {
        if let Some(t) = *temperature {
            require_temperature(t, context)?;
            if e.is_none() {
                return Err(PyValueError::new_err(format!(
                    "{context} requires None as a continuous factor's temperature"
                )));
            }
        }
    }
    let sizes = u_weights
        .iter()
        .map(|u| cell_size(u, context))
        .collect::<PyResult<Vec<_>>>()?;
    let (starts, total) = factor_starts(&sizes, context)?;
    if cross && sizes.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "{context} requires 2 or more factors for the cross-term, got {}",
            sizes.len()
        )));
    }
    let (rows, width) = matrix_shape(xs[0], "x", context)?;
    for x in xs {
        require_shape(x, Shape::Matrix(rows, width), "each x", context)?;
    }
    require_shape(c, Shape::Matrix(rows, total), "c", context)?;
    let features = features_per_token(tokens, width, context)?;

    let mut factors = Vec::with_capacity(sizes.len());
    for (g, c_g) in split(c, total, &starts, &sizes).into_iter().enumerate() {
        let (u, p) = attention_pooling_forward(xs[g], queries[g], tokens)?;
        let input_size = features + if cross { total - sizes[g] } else { 0 };
        let input = if cross {
            join(&[u, others(c, total, starts[g], sizes[g])], rows, input_size)
        } else {
            u
        };
        require_shape(w[g], Shape::Matrix(3 * sizes[g], input_size), "each w", context)?;
        let (h, r, z, h_tilde, rc) = gru_forward(&input, &c_g, w[g], u_weights[g], b[g])?;
        let (q, indices, soft) = snap(&h, codes[g], temperatures[g])?;
        factors.push(FactorTape {
            p,
            input,
            c: c_g,
            r,
            z,
            h_tilde,
            rc,
            h,
            q,
            indices,
            soft,
        });
    }
    let qs: Vec<RustArray> = factors.iter().map(|f| f.q.clone()).collect();
    let hs: Vec<RustArray> = factors.iter().map(|f| f.h.clone()).collect();
    Ok(StepCore {
        state: join(&qs, rows, total),
        h: join(&hs, rows, total),
        tokens,
        cross,
        features,
        rows,
        sizes,
        starts,
        factors,
    })
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
/// `temperatures[g]` (`soft_codes_forward`; None, or no `temperatures`, hard). Returns the
/// `RecurrenceStep`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (x, c, queries, w, u_weights, b, codes, tokens, *, cross, temperatures = None))]
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
) -> PyResult<RecurrenceStep> {
    let codes: Vec<Option<&RustArray>> = codes.iter().map(|e| e.as_deref()).collect();
    let temperatures = temperatures.unwrap_or_else(|| vec![None; codes.len()]);
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
        tokens,
        cross,
        "recurrence_step_forward",
    )?;
    drop(borrowed);
    let mut core = core;
    let state = std::mem::replace(&mut core.state, RustArray::from_vector(Vec::new()));
    let h = std::mem::replace(&mut core.h, RustArray::from_vector(Vec::new()));
    Ok(RecurrenceStep {
        state: Py::new(py, state)?,
        h: Py::new(py, h)?,
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
/// state's, and each factor's parameters'.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (step, dq, queries, w, u_weights, grad_queries, grad_w, grad_u, grad_b, grad_codes, weight, commitments, targets = None))]
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
    let codes: Vec<Option<&RustArray>> = grad_codes.iter().map(|e| e.as_deref()).collect();
    let (dxs, dc, gq, gw, gu, gb, ge) = step.core.backward(
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
        "recurrence_step_backward",
    )?;
    let dx = match &step.x {
        StepTokens::Shared(_) => {
            let all: Vec<usize> = (0..dxs.len()).collect();
            Py::new(py, sum_in_order(&dxs, &all).expect("a step has 1 or more factors"))?.into_any()
        }
        StepTokens::Each(_) => dxs
            .into_iter()
            .map(|dx| Py::new(py, dx))
            .collect::<PyResult<Vec<_>>>()?
            .into_pyobject(py)?
            .into_any()
            .unbind(),
    };
    Ok((dx, dc, gq, gw, gu, gb, ge))
}

/// The parameters' gradients a step's backward pass accumulates onto, each factor's.
pub(crate) struct StepGradients<'a> {
    pub(crate) queries: &'a [&'a RustArray],
    pub(crate) w: &'a [&'a RustArray],
    pub(crate) u: &'a [&'a RustArray],
    pub(crate) b: &'a [&'a RustArray],
    pub(crate) codes: &'a [Option<&'a RustArray>],
}

/// `StepCore::backward`'s result: `(dxs, dc, grad_queries, grad_w, grad_u, grad_b, grad_codes)`,
/// `dxs` each factor's pooling's gradient of its tokens.
pub(crate) type StepBackward = (
    Vec<RustArray>,
    RustArray,
    Vec<RustArray>,
    Vec<RustArray>,
    Vec<RustArray>,
    Vec<RustArray>,
    Vec<Option<RustArray>>,
);

impl StepCore {
    /// `recurrence_step_backward`'s computation, from the tokens `xs` the step read (one a factor),
    /// under `context`'s name; each factor's commitment target `targets[g]` (None: its `q`). Its
    /// tokens' gradient is each factor's pooling's, apart (`sum_in_order` sums those that read the
    /// same tokens).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn backward(
        &self,
        xs: &[&RustArray],
        dq: &RustArray,
        queries: &[&RustArray],
        w: &[&RustArray],
        u_weights: &[&RustArray],
        grads: &StepGradients<'_>,
        weight: f64,
        commitments: &[f64],
        targets: &[Option<&RustArray>],
        context: &str,
    ) -> PyResult<StepBackward> {
        let count = factor_count(
            &[
                (self.sizes.len(), "the step's factors"),
                (xs.len(), "x"),
                (queries.len(), "queries"),
                (w.len(), "w"),
                (u_weights.len(), "u_weights"),
                (grads.queries.len(), "grad_queries"),
                (grads.w.len(), "grad_w"),
                (grads.u.len(), "grad_u"),
                (grads.b.len(), "grad_b"),
                (grads.codes.len(), "grad_codes"),
                (commitments.len(), "commitments"),
                (targets.len(), "targets"),
            ],
            context,
        )?;
        let rows = self.rows;
        let total = self.starts[count - 1] + self.sizes[count - 1];
        require_shape(dq, Shape::Matrix(rows, total), "dq", context)?;
        let features = self.features;

        let mut dxs = Vec::with_capacity(count);
        let mut dc_own = Vec::with_capacity(count);
        let mut dc_cross = Vec::with_capacity(count);
        let mut new_grad_q = Vec::with_capacity(count);
        let mut new_grad_w = Vec::with_capacity(count);
        let mut new_grad_u = Vec::with_capacity(count);
        let mut new_grad_b = Vec::with_capacity(count);
        let mut new_grad_codes = Vec::with_capacity(count);
        for (g, dq_g) in split(dq, total, &self.starts, &self.sizes).into_iter().enumerate() {
            let f = &self.factors[g];
            let (dh, grad_e) = snap_backward(
                dq_g,
                &f.h,
                &f.q,
                &f.indices,
                f.soft.as_ref(),
                grads.codes[g],
                weight,
                commitments[g],
                targets[g],
                context,
            )?;
            let (d_input, dc, gw, gu, gb) = gru_backward(
                &dh,
                &f.input,
                &f.c,
                &f.r,
                &f.z,
                &f.h_tilde,
                &f.rc,
                w[g],
                u_weights[g],
                grads.w[g],
                grads.u[g],
                grads.b[g],
            )?;
            let input_size = features + if self.cross { total - self.sizes[g] } else { 0 };
            let du = if self.cross {
                dc_cross.push(columns(&d_input, input_size, features, input_size));
                columns(&d_input, input_size, 0, features)
            } else {
                d_input
            };
            let (dx_g, gq) = attention_pooling_backward(xs[g], &f.p, queries[g], &du, grads.queries[g], self.tokens)?;
            dxs.push(dx_g);
            dc_own.push(dc);
            new_grad_q.push(gq);
            new_grad_w.push(gw);
            new_grad_u.push(gu);
            new_grad_b.push(gb);
            new_grad_codes.push(grad_e);
        }

        let mut dc = join(&dc_own, rows, total);
        // each factor h's gradient through the other factors' columns, c_(-h), in factor order
        for (h, cross) in dc_cross.iter().enumerate() {
            let (start, size) = (self.starts[h], self.sizes[h]);
            let width = total - size;
            for (row, values) in dc.data.chunks_exact_mut(total).zip(cross.data.chunks_exact(width)) {
                let (before, rest) = row.split_at_mut(start);
                let targets = before.iter_mut().chain(rest[size..].iter_mut());
                for (acc, &v) in targets.zip(values) {
                    *acc += v;
                }
            }
        }
        Ok((dxs, dc, new_grad_q, new_grad_w, new_grad_u, new_grad_b, new_grad_codes))
    }
}

/// The poolings' gradients `dxs[g]` of the factors `factors` (indices into `dxs`, in factor order)
/// summed in that order, as one step's backward pass summed every factor's before the window
/// program's branches: `dxs[f_0] + dxs[f_1] + ...`, left to right. None for no factors.
pub(crate) fn sum_in_order(dxs: &[RustArray], factors: &[usize]) -> Option<RustArray> {
    let mut sum: Option<RustArray> = None;
    for &g in factors {
        let dx = &dxs[g];
        sum = Some(match sum {
            None => dx.clone(),
            Some(acc) => RustArray {
                data: same_shape_elementwise(&acc.data, &dx.data, |a, v| a + v),
                shape: acc.shape,
            },
        });
    }
    sum
}
