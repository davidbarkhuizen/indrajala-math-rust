//! A factored window-recurrent state (indrajala-ml's factored-state workplan, stage 2): G factors,
//! each its own pooling, GRU cell and codes, the state `(N, S)` their states side by side, factor
//! `g` the `s_g` columns after the factors before it (`S = s_1 + ... + s_G`).
//!
//! ```text
//! pooling g     u_g = pool(x; q_g)                       (attention_pooling_forward)
//! cell g        h_g = GRU_g(c_g, i_g),  i_g = u_g, or [u_g, c_(-g)] with the cross-term on
//! codes g       q_g = e_(g, k), k = argmin ||h_g - e_g,k||^2   (codes_forward); q_g = h_g if continuous
//! coupling g    q'_g = (1 - l_g) h_g + l_g q_g, l_g in [0, 1]  (1: q_g itself, today's by bits)
//! next state    [q'_1, ..., q'_G]
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

use crate::array::{Array, Shape};
use crate::ops::same_shape_elementwise;
use crate::recurrent::{
    attention_pooling_backward, attention_pooling_forward, codes_backward, codes_forward, columns, features_per_token,
    gru_backward, gru_forward, matrix_shape, require_shape, require_temperature, soft_codes_backward,
    soft_codes_forward,
};
use crate::Error;

/// Each factor's first column, from the factors' sizes, every one 1 or more; and their total.
fn factor_starts(sizes: &[usize], context: &str) -> Result<(Vec<usize>, usize), Error> {
    if sizes.is_empty() || sizes.contains(&0) {
        return Err(Error::invalid(format!(
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
fn split(x: &Array, total: usize, starts: &[usize], sizes: &[usize]) -> Vec<Array> {
    starts
        .iter()
        .zip(sizes)
        .map(|(&start, &size)| columns(x, total, start, start + size))
        .collect()
}

/// The factors' blocks side by side, `(rows, total)`: `split`'s inverse.
fn join(blocks: &[Array], rows: usize, total: usize) -> Array {
    let mut data = Vec::with_capacity(rows * total);
    for row in 0..rows {
        for block in blocks {
            let width = block.data.len() / rows.max(1);
            data.extend_from_slice(&block.data[row * width..(row + 1) * width]);
        }
    }
    Array::from_matrix(data, rows, total)
}

/// The list arguments' common length, the factor count G, refused unless they agree.
fn factor_count(lengths: &[(usize, &str)], context: &str) -> Result<usize, Error> {
    let (count, _) = lengths[0];
    if let Some((length, name)) = lengths.iter().find(|(length, _)| *length != count) {
        return Err(Error::invalid(format!(
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
    weights: Array,
    /// the codes the step read, a copy, `(m_g, s_g)`
    codes: Array,
}

/// Factor `g`'s state `c_g` snapped to its nearest code, or with a temperature its soft state
/// (`soft_codes_forward`), or itself for a continuous factor: `(q, indices, soft)`, the indices
/// the nearest codes either way.
#[allow(clippy::type_complexity)]
fn snap(
    h: &Array,
    codes: Option<&Array>,
    temperature: Option<f64>,
) -> Result<(Array, Option<Vec<usize>>, Option<Soft>), Error> {
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
        (None, Some(_)) => Err(Error::invalid(
            "a continuous factor has no codes to soften: its temperature must be None",
        )),
    }
}

/// `snap`'s backward pass: `codes_backward` (with the commitment's `target`, if any), or
/// `soft_codes_backward` for a soft factor (no target; `straight`, its gradient straight through),
/// or `dq` itself for a continuous factor. Returns `(dh, grad_e)`.
#[allow(clippy::too_many_arguments)]
fn snap_backward(
    dq: Array,
    h: &Array,
    q: &Array,
    indices: &Option<Vec<usize>>,
    soft: Option<&Soft>,
    grad_e: Option<&Array>,
    weight: f64,
    commitment: f64,
    target: Option<&Array>,
    straight: bool,
    context: &str,
) -> Result<(Array, Option<Array>), Error> {
    if let (Some(soft), Some(grad_e)) = (soft, grad_e) {
        if target.is_some() {
            return Err(Error::invalid(format!(
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
            straight,
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
            Some(_) => Err(Error::invalid(format!(
                "{context} requires None as a continuous factor's commitment target"
            ))),
        },
        (Some(_), None) => Err(Error::invalid(format!(
            "{context} requires a codes gradient for each snapped factor, got None"
        ))),
        (None, Some(_)) => Err(Error::invalid(format!(
            "{context} requires None as a continuous factor's codes gradient"
        ))),
    }
}

/// Each row of `h`, `(N, S)`, snapped factor by factor: factor `g`'s `sizes[g]` columns to the
/// nearest of its codes `codes[g]` (`codes_forward`), or kept for a continuous factor (None).
/// Returns `(q, indices)`, `q` `(N, S)` and each factor's indices, None for a continuous factor.
#[allow(clippy::type_complexity)]
pub fn factored_codes_forward(
    h: &Array,
    sizes: Vec<usize>,
    codes: &[Option<&Array>],
) -> Result<(Array, Vec<Option<Vec<usize>>>), Error> {
    let context = "factored_codes_forward";
    factor_count(&[(sizes.len(), "sizes"), (codes.len(), "codes")], context)?;
    let (starts, total) = factor_starts(&sizes, context)?;
    let (rows, _) = matrix_shape(h, "h", context)?;
    require_shape(h, Shape::Matrix(rows, total), "h", context)?;
    let mut blocks = Vec::with_capacity(sizes.len());
    let mut indices = Vec::with_capacity(sizes.len());
    for (h_g, e) in split(h, total, &starts, &sizes).iter().zip(codes) {
        let (q, i, _) = snap(h_g, *e, None)?;
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
pub fn factored_codes_backward(
    dq: &Array,
    h: &Array,
    q: &Array,
    sizes: Vec<usize>,
    indices: Vec<Option<Vec<usize>>>,
    grad_codes: &[Option<&Array>],
    weight: f64,
    commitments: Vec<f64>,
    targets: Option<&[Option<&Array>]>,
) -> Result<(Array, Vec<Option<Array>>), Error> {
    let context = "factored_codes_backward";
    let targets = targets.map_or_else(|| vec![None; sizes.len()], <[_]>::to_vec);
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
            grad_codes[g],
            weight,
            commitments[g],
            targets[g],
            false,
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
    p: Array,
    /// the cell's input `i_g`, `(N, d)` or `(N, d + S - s_g)`
    input: Array,
    /// the factor's state `c_g`, `(N, s_g)`
    c: Array,
    r: Array,
    z: Array,
    h_tilde: Array,
    rc: Array,
    /// the cell's output before it is snapped, `(N, s_g)`
    h: Array,
    /// the next state's block, `(N, s_g)`
    q: Array,
    indices: Option<Vec<usize>>,
    /// a soft factor's weights and codes, None if hard or continuous
    soft: Option<Soft>,
    /// the coupling `l_g`: the next state's share of the snapped `q`, the rest the cell's `h`
    coupling: f64,
    /// the coupled next state `(1 - l_g) h + l_g q`, None at coupling 1 (the next state is `q`)
    coupled: Option<Array>,
}

/// A factor's coupled next state, `(1 - coupling) h + coupling q`, element by element.
fn couple(h: &Array, q: &Array, coupling: f64) -> Array {
    let rest = 1.0 - coupling;
    Array {
        data: same_shape_elementwise(&h.data, &q.data, |h, q| rest * h + coupling * q),
        shape: h.shape,
    }
}

/// Each factor's coupling checked: one a factor, in `[0, 1]`, below 1 only for a factor with codes.
fn check_couplings(couplings: &[f64], codes: &[Option<&Array>], context: &str) -> Result<(), Error> {
    for (g, (&coupling, e)) in couplings.iter().zip(codes).enumerate() {
        if !(0.0..=1.0).contains(&coupling) {
            return Err(Error::invalid(format!(
                "{context} requires each coupling in [0, 1], factor {g}'s is {coupling}"
            )));
        }
        if coupling < 1.0 && e.is_none() {
            return Err(Error::invalid(format!(
                "{context} couples only a factor with codes (not a continuous one) below 1, factor {g}"
            )));
        }
    }
    Ok(())
}

/// Factor `g`'s state size `s_g` from its `u_weights`, `(3 s_g, s_g)`.
fn cell_size(u_weights: &Array, context: &str) -> Result<usize, Error> {
    let (rows, size) = matrix_shape(u_weights, "every u_weights", context)?;
    if rows != 3 * size || size == 0 {
        return Err(Error::invalid(format!(
            "{context} requires each u_weights of shape (3s, s), s >= 1, got {:?}",
            u_weights.shape
        )));
    }
    Ok(size)
}

/// The columns of `c` that aren't factor `g`'s, in factor order, `(N, S - s_g)`.
fn others(c: &Array, total: usize, start: usize, size: usize) -> Array {
    let rows = c.data.len() / total;
    let mut data = Vec::with_capacity(rows * (total - size));
    for row in c.data.chunks_exact(total) {
        data.extend_from_slice(&row[..start]);
        data.extend_from_slice(&row[start + size..]);
    }
    Array::from_matrix(data, rows, total - size)
}

/// A window's recurrence step, as `recurrence_step_forward` computes it, without Python's
/// references: what `RecurrenceStep` holds, and the window program's step (program.rs).
pub struct StepCore {
    tokens: usize,
    cross: bool,
    pub features: usize,
    pub rows: usize,
    pub sizes: Vec<usize>,
    pub starts: Vec<usize>,
    factors: Vec<FactorTape>,
    /// the next state, `(N, S)`
    pub state: Array,
    /// the cells' outputs before they are snapped, `(N, S)`
    pub h: Array,
}

impl StepCore {
    /// Whether each factor's state was snapped to its codes (hard or soft) in this step: false for
    /// a continuous factor, whose codes take no gradient.
    pub fn snapped(&self) -> Vec<bool> {
        self.factors
            .iter()
            .map(|f| f.indices.is_some() || f.soft.is_some())
            .collect()
    }

    /// A step of no factors, a placeholder until `step_forward` gives one.
    pub fn empty() -> StepCore {
        StepCore {
            tokens: 0,
            cross: false,
            features: 0,
            rows: 0,
            sizes: Vec::new(),
            starts: Vec::new(),
            factors: Vec::new(),
            state: Array::from_vector(Vec::new()),
            h: Array::from_vector(Vec::new()),
        }
    }

    /// How many values the step keeps for its backward pass.
    pub fn floats(&self) -> usize {
        let tapes: usize = self
            .factors
            .iter()
            .map(|f| {
                [&f.p, &f.input, &f.c, &f.r, &f.z, &f.h_tilde, &f.rc, &f.h, &f.q]
                    .iter()
                    .map(|a| a.data.len())
                    .sum::<usize>()
                    + f.soft.as_ref().map_or(0, |s| s.weights.data.len() + s.codes.data.len())
                    + f.coupled.as_ref().map_or(0, |a| a.data.len())
            })
            .sum();
        tapes + self.state.data.len() + self.h.data.len()
    }

    /// Each factor's code indices, one per example; None for a continuous factor.
    pub fn indices(&self) -> Vec<Option<Vec<usize>>> {
        self.factors.iter().map(|f| f.indices.clone()).collect()
    }
}

/// Codes `e`'s `(count, size)`; a vector, which no factor's codes are, `(0, 0)`.
fn code_shape(e: &Array) -> (usize, usize) {
    match e.shape {
        Shape::Matrix(count, size) => (count, size),
        Shape::Vector(_) => (0, 0),
    }
}

/// The consolidation table (indrajala-ml's code-consolidation workplan, D1): each duplicate joint
/// code, a code index per factor, to its representative's, checked against the factors' `codes`
/// (every factor snapped, every index one of its factor's codes, each duplicate once).
pub struct Consolidation {
    table: HashMap<Vec<usize>, Vec<usize>>,
}

impl Consolidation {
    pub fn new(
        pairs: Vec<(Vec<usize>, Vec<usize>)>,
        codes: &[Option<&Array>],
        context: &str,
    ) -> Result<Consolidation, Error> {
        if !pairs.is_empty() && codes.iter().any(Option::is_none) {
            return Err(Error::invalid(format!(
                "{context} consolidates only when every factor is snapped: a continuous factor has no codes"
            )));
        }
        let counts: Vec<usize> = codes.iter().map(|e| e.map_or(0, |e| code_shape(e).0)).collect();
        let mut table = HashMap::with_capacity(pairs.len());
        for (duplicate, representative) in pairs {
            for code in [&duplicate, &representative] {
                if code.len() != counts.len() || code.iter().zip(&counts).any(|(&i, &m)| i >= m) {
                    return Err(Error::invalid(format!(
                        "{context} requires each consolidation code as a code index per factor, below its factor's \
                         code counts {counts:?}, got {code:?}"
                    )));
                }
            }
            if let Some(earlier) = table.insert(duplicate.clone(), representative) {
                return Err(Error::invalid(format!(
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
    pub fn targets(&self, step: &StepCore, codes: &[Option<&Array>]) -> Vec<Option<Array>> {
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
                Some(Array::from_matrix(data, rows, size))
            })
            .collect()
    }
}

/// `recurrence_step_forward`'s computation over plain references, under `context`'s name: factor
/// `g`'s pooling reads `xs[g]` (the same tokens for every factor, or each its own: the window
/// program's branches), every one of the same shape; factor `g` soft with `temperatures[g]`
/// (None: hard), its next state coupled by `couplings[g]` (1: its `q`).
#[allow(clippy::too_many_arguments)]
pub fn step_forward(
    xs: &[&Array],
    c: &Array,
    queries: &[&Array],
    w: &[&Array],
    u_weights: &[&Array],
    b: &[&Array],
    codes: &[Option<&Array>],
    temperatures: &[Option<f64>],
    couplings: &[f64],
    tokens: usize,
    cross: bool,
    context: &str,
) -> Result<StepCore, Error> {
    factor_count(
        &[
            (xs.len(), "x"),
            (queries.len(), "queries"),
            (w.len(), "w"),
            (u_weights.len(), "u_weights"),
            (b.len(), "b"),
            (codes.len(), "codes"),
            (temperatures.len(), "temperatures"),
            (couplings.len(), "couplings"),
        ],
        context,
    )?;
    check_couplings(couplings, codes, context)?;
    for (temperature, e) in temperatures.iter().zip(codes) {
        if let Some(t) = *temperature {
            require_temperature(t, context)?;
            if e.is_none() {
                return Err(Error::invalid(format!(
                    "{context} requires None as a continuous factor's temperature"
                )));
            }
        }
    }
    let sizes = u_weights
        .iter()
        .map(|u| cell_size(u, context))
        .collect::<Result<Vec<_>, Error>>()?;
    let (starts, total) = factor_starts(&sizes, context)?;
    if cross && sizes.len() < 2 {
        return Err(Error::invalid(format!(
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
        let coupling = couplings[g];
        let coupled = (coupling < 1.0).then(|| couple(&h, &q, coupling));
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
            coupling,
            coupled,
        });
    }
    let qs: Vec<Array> = factors
        .iter()
        .map(|f| f.coupled.as_ref().unwrap_or(&f.q).clone())
        .collect();
    let hs: Vec<Array> = factors.iter().map(|f| f.h.clone()).collect();
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

/// The parameters' gradients a step's backward pass accumulates onto, each factor's.
pub struct StepGradients<'a> {
    pub queries: &'a [&'a Array],
    pub w: &'a [&'a Array],
    pub u: &'a [&'a Array],
    pub b: &'a [&'a Array],
    pub codes: &'a [Option<&'a Array>],
}

/// `StepCore::backward`'s result: `(dxs, dc, grad_queries, grad_w, grad_u, grad_b, grad_codes)`,
/// `dxs` each factor's pooling's gradient of its tokens.
pub type StepBackward = (
    Vec<Array>,
    Array,
    Vec<Array>,
    Vec<Array>,
    Vec<Array>,
    Vec<Array>,
    Vec<Option<Array>>,
);

impl StepCore {
    /// `recurrence_step_backward`'s computation, from the tokens `xs` the step read (one a factor),
    /// under `context`'s name; each factor's commitment target `targets[g]` (None: its `q`). Its
    /// tokens' gradient is each factor's pooling's, apart (`sum_in_order` sums those that read the
    /// same tokens). With `straight`, a soft factor's gradient is straight through
    /// (`soft_codes_backward`'s).
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        xs: &[&Array],
        dq: &Array,
        queries: &[&Array],
        w: &[&Array],
        u_weights: &[&Array],
        grads: &StepGradients<'_>,
        weight: f64,
        commitments: &[f64],
        targets: &[Option<&Array>],
        straight: bool,
        context: &str,
    ) -> Result<StepBackward, Error> {
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
            // coupled: the snapped path takes l_g dq (its codebook and commitment terms whole), the
            // cell's output (1 - l_g) dq beside it
            let rest = (f.coupled.is_some()).then(|| (1.0 - f.coupling, dq_g.clone()));
            let through = match rest {
                Some(_) => Array {
                    data: dq_g.data.iter().map(|d| f.coupling * d).collect(),
                    shape: dq_g.shape,
                },
                None => dq_g,
            };
            let (dh, grad_e) = snap_backward(
                through,
                &f.h,
                &f.q,
                &f.indices,
                f.soft.as_ref(),
                grads.codes[g],
                weight,
                commitments[g],
                targets[g],
                straight,
                context,
            )?;
            let dh = match rest {
                Some((share, dq_g)) => Array {
                    data: same_shape_elementwise(&dh.data, &dq_g.data, |a, d| a + share * d),
                    shape: dh.shape,
                },
                None => dh,
            };
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
pub fn sum_in_order(dxs: &[Array], factors: &[usize]) -> Option<Array> {
    let mut sum: Option<Array> = None;
    for &g in factors {
        let dx = &dxs[g];
        sum = Some(match sum {
            None => dx.clone(),
            Some(acc) => Array {
                data: same_shape_elementwise(&acc.data, &dx.data, |a, v| a + v),
                shape: acc.shape,
            },
        });
    }
    sum
}
