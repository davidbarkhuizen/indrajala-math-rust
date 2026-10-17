//! The window program (indrajala-ml's window-program workplan, stage 2): a window-recurrent
//! network's window, its layers as a list of ops, run over every window of an input in one call,
//! each window's layers then its factored recurrence step.
//!
//! ```text
//! window w      a_0 = x[:, w k : (w + 1) k],  a_(i+1) = op_i(a_i)   (the StateInputs read c_w)
//! step w        c_(w+1) = step(a_reads, c_w)                          (factored.rs's step_forward)
//! outputs       [a_L for each window], side by side
//! ```
//!
//! Each op calls the crate function the indrajala-ml layer of its kind calls, with the same
//! operands in the same order, and the compositions that layer does in Python (the residual's add,
//! Position's add, a token-wise dense layer's reshapes) are the same `RustArray` operations, so a
//! program's outputs, states and code indices are the per-layer path's by bits (the workplan's D3).
//! The parameters come with each call, one flat list in the network's parameter order (D2): each
//! op's in turn, then each factor's pooling query, cell `w`, `u`, `b`, codes (if snapped) and
//! initial state (which the program doesn't read: the caller snaps the initial state).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::array::{RustArray, Shape};
use crate::attention::attention_forward_batch;
use crate::factored::step_forward;
use crate::fused::{affine_forward_batch, layer_relu_forward_batch, layer_softmax_forward_batch};
use crate::layer_norm::layer_norm_forward_batch;
use crate::recurrent::{columns, join_columns, matrix_shape, state_input_forward};
use crate::tokens::embedding_forward;

/// One layer of a window, as the program runs it.
#[derive(Clone)]
enum Op {
    /// `embedding_forward`; parameters `E`
    Embedding,
    /// the tokens plus `P` as one row; parameters `P` `(T, d)`
    Position,
    /// `state_input_forward` with the window's state; parameters `P`
    StateInput { tokens: usize },
    /// a residual block opens: its input kept for its add
    Fork,
    /// a residual block closes: the body's output plus the block's input
    Add,
    /// `layer_norm_forward_batch`; parameters `gamma`, `beta`
    LayerNorm { epsilon: f64 },
    /// a dense layer on each token's row, ReLU or affine; parameters `W`, `b`
    TokenDense { tokens: usize, relu: bool },
    /// the token-wise softmax output layer; parameters `W`, `b`
    TokenOutput { tokens: usize },
    /// `attention_forward_batch`; parameters `Wq`, `bq`, `Wk`, `bk`, `Wv`, `bv`, `Wo`, `bo`
    Attention {
        heads: usize,
        causal: bool,
        dropout: f64,
        cos: Option<RustArray>,
        sin: Option<RustArray>,
    },
    /// token-wise dropout: nothing in inference
    Dropout,
}

impl Op {
    fn parameter_count(&self) -> usize {
        match self {
            Op::Embedding | Op::Position | Op::StateInput { .. } => 1,
            Op::Fork | Op::Add | Op::Dropout => 0,
            Op::LayerNorm { .. } | Op::TokenDense { .. } | Op::TokenOutput { .. } => 2,
            Op::Attention { .. } => 8,
        }
    }

    /// The op a descriptor names: `(kind, *settings)`, as `WindowProgram`'s documentation lists.
    fn parse(descriptor: &Bound<'_, PyTuple>) -> PyResult<Op> {
        let context = "WindowProgram";
        let kind: String = descriptor.get_item(0)?.extract()?;
        let settings = descriptor.len() - 1;
        let expect = |count: usize| -> PyResult<()> {
            if settings == count {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!(
                    "{context}'s {kind:?} op takes {count} settings, got {settings}"
                )))
            }
        };
        let item = |i: usize| descriptor.get_item(i);
        Ok(match kind.as_str() {
            "embedding" => {
                expect(0)?;
                Op::Embedding
            }
            "position" => {
                expect(0)?;
                Op::Position
            }
            "state_input" => {
                expect(1)?;
                Op::StateInput {
                    tokens: item(1)?.extract()?,
                }
            }
            "fork" => {
                expect(0)?;
                Op::Fork
            }
            "add" => {
                expect(0)?;
                Op::Add
            }
            "layer_norm" => {
                expect(1)?;
                Op::LayerNorm {
                    epsilon: item(1)?.extract()?,
                }
            }
            "token_dense" => {
                expect(2)?;
                let activation: String = item(2)?.extract()?;
                if activation != "relu" && activation != "linear" {
                    return Err(PyValueError::new_err(format!(
                        "{context}'s token_dense activation is \"relu\" or \"linear\", got {activation:?}"
                    )));
                }
                Op::TokenDense {
                    tokens: item(1)?.extract()?,
                    relu: activation == "relu",
                }
            }
            "token_output" => {
                expect(1)?;
                Op::TokenOutput {
                    tokens: item(1)?.extract()?,
                }
            }
            "attention" => {
                expect(5)?;
                let table = |i: usize| -> PyResult<Option<RustArray>> {
                    let value = item(i)?;
                    if value.is_none() {
                        Ok(None)
                    } else {
                        Ok(Some(value.extract::<PyRef<'_, RustArray>>()?.clone()))
                    }
                };
                let (cos, sin) = (table(4)?, table(5)?);
                if cos.is_some() != sin.is_some() {
                    return Err(PyValueError::new_err(format!(
                        "{context}'s attention op takes both cos and sin, or neither"
                    )));
                }
                Op::Attention {
                    heads: item(1)?.extract()?,
                    causal: item(2)?.extract()?,
                    dropout: item(3)?.extract()?,
                    cos,
                    sin,
                }
            }
            "dropout" => {
                expect(1)?;
                let _: f64 = item(1)?.extract()?;
                Op::Dropout
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "{context} has no op {kind:?}: embedding, position, state_input, fork, add, layer_norm, \
                     token_dense, token_output, attention, dropout"
                )))
            }
        })
    }
}

/// A window-recurrent network's window as ops, and its recurrence's shape: what
/// `window_program_forward` runs.
///
/// `ops` are tuples, a kind then its settings: `("embedding",)`, `("position",)`,
/// `("state_input", tokens)`, `("fork",)`, `("add",)`, `("layer_norm", epsilon)`,
/// `("token_dense", tokens, "relu" | "linear")`, `("token_output", tokens)`,
/// `("attention", heads, causal, dropout, cos, sin)` (`cos`, `sin` the rotary table's arrays, or
/// None for both), `("dropout", p)`. `columns` is a window's width `k` (its tokens), `reads` the
/// op whose input the poolings read, `sizes` the factors' state sizes, `snapped` whether each has
/// codes, `cross` the cross-term.
#[pyclass(module = "indrajala_math_rust", frozen, skip_from_py_object)]
pub struct WindowProgram {
    ops: Vec<Op>,
    columns: usize,
    reads: usize,
    sizes: Vec<usize>,
    snapped: Vec<bool>,
    cross: bool,
}

#[pymethods]
impl WindowProgram {
    #[new]
    #[pyo3(signature = (ops, columns, reads, sizes, snapped, *, cross))]
    fn new(
        ops: Vec<Bound<'_, PyTuple>>,
        columns: usize,
        reads: usize,
        sizes: Vec<usize>,
        snapped: Vec<bool>,
        cross: bool,
    ) -> PyResult<Self> {
        let context = "WindowProgram";
        let ops = ops.iter().map(Op::parse).collect::<PyResult<Vec<_>>>()?;
        if ops.is_empty() || reads >= ops.len() {
            return Err(PyValueError::new_err(format!(
                "{context} requires 1 or more ops and reads among them, got {} ops and reads {reads}",
                ops.len()
            )));
        }
        if columns == 0 || sizes.is_empty() || sizes.len() != snapped.len() || sizes.contains(&0) {
            return Err(PyValueError::new_err(format!(
                "{context} requires columns >= 1 and one snapped flag per factor size, each size 1 or more, \
                 got columns {columns}, sizes {sizes:?}, snapped {snapped:?}"
            )));
        }
        Ok(WindowProgram {
            ops,
            columns,
            reads,
            sizes,
            snapped,
            cross,
        })
    }

    /// How many parameters a call takes: the ops', then the factors'.
    #[getter]
    fn parameter_count(&self) -> usize {
        self.op_parameters() + self.snapped.iter().map(|&s| 5 + usize::from(s)).sum::<usize>()
    }
}

/// One window's ops' parameters and the recurrence's, split from a call's list.
struct Parameters<'a> {
    ops: &'a [&'a RustArray],
    queries: Vec<&'a RustArray>,
    w: Vec<&'a RustArray>,
    u: Vec<&'a RustArray>,
    b: Vec<&'a RustArray>,
    codes: Vec<Option<&'a RustArray>>,
}

impl WindowProgram {
    fn op_parameters(&self) -> usize {
        self.ops.iter().map(Op::parameter_count).sum()
    }

    fn split<'a>(&self, parameters: &'a [&'a RustArray], context: &str) -> PyResult<Parameters<'a>> {
        let expected = self.parameter_count();
        if parameters.len() != expected {
            return Err(PyValueError::new_err(format!(
                "{context} requires {expected} parameters, the ops' then each factor's, got {}",
                parameters.len()
            )));
        }
        let count = self.op_parameters();
        let mut split = Parameters {
            ops: &parameters[..count],
            queries: Vec::new(),
            w: Vec::new(),
            u: Vec::new(),
            b: Vec::new(),
            codes: Vec::new(),
        };
        let mut at = count;
        for &snapped in &self.snapped {
            split.queries.push(parameters[at]);
            split.w.push(parameters[at + 1]);
            split.u.push(parameters[at + 2]);
            split.b.push(parameters[at + 3]);
            at += 4;
            split.codes.push(if snapped { Some(parameters[at]) } else { None });
            at += usize::from(snapped) + 1; // the codes, then the initial state
        }
        Ok(split)
    }

    /// One window's ops from its columns `x` and its state `c`: its output and the tokens the
    /// poolings read.
    fn window(&self, x: RustArray, c: &RustArray, parameters: &[&RustArray]) -> PyResult<(RustArray, RustArray)> {
        let mut a = x;
        let mut forks: Vec<RustArray> = Vec::new();
        let mut read = None;
        let mut at = 0;
        for (i, op) in self.ops.iter().enumerate() {
            if i == self.reads {
                read = Some(a.clone());
            }
            let p = &parameters[at..at + op.parameter_count()];
            at += op.parameter_count();
            a = match op {
                Op::Embedding => embedding_forward(&a, p[0])?,
                Op::Position => {
                    a.combine_with_array(&RustArray::from_vector(p[0].data.clone()), |x, y| x + y, "add")?
                }
                Op::StateInput { tokens } => state_input_forward(&a, c, p[0], *tokens)?,
                Op::Fork => {
                    forks.push(a.clone());
                    a
                }
                Op::Add => {
                    let Some(skip) = forks.pop() else {
                        return Err(PyValueError::new_err("window_program_forward: an add without its fork"));
                    };
                    a.combine_with_array(&skip, |x, y| x + y, "add")?
                }
                Op::LayerNorm { epsilon } => layer_norm_forward_batch(&a, p[0], p[1], *epsilon)?.0,
                Op::TokenDense { tokens, relu } => {
                    let forward = if *relu {
                        layer_relu_forward_batch
                    } else {
                        affine_forward_batch
                    };
                    token_rows(&a, *tokens, p[0], p[1], forward)?
                }
                Op::TokenOutput { tokens } => token_rows(&a, *tokens, p[0], p[1], layer_softmax_forward_batch)?,
                Op::Attention {
                    heads,
                    causal,
                    dropout,
                    cos,
                    sin,
                } => {
                    attention_forward_batch(
                        &a,
                        p[0],
                        p[1],
                        p[2],
                        p[3],
                        p[4],
                        p[5],
                        p[6],
                        p[7],
                        *heads,
                        *causal,
                        *dropout,
                        false,
                        None,
                        cos.as_ref(),
                        sin.as_ref(),
                    )?
                    .0
                }
                Op::Dropout => a,
            };
        }
        Ok((a, read.expect("reads is among the ops")))
    }
}

/// A token-wise dense layer's pass: `a` `(N, T * in)` as `(N * T, in)` rows, `forward(w, rows, b)`,
/// back as `(N, T * size)`.
fn token_rows(
    a: &RustArray,
    tokens: usize,
    w: &RustArray,
    b: &RustArray,
    forward: fn(&RustArray, &RustArray, &RustArray) -> PyResult<RustArray>,
) -> PyResult<RustArray> {
    let context = "window_program_forward";
    let (examples, _) = matrix_shape(a, "a token-wise layer's input", context)?;
    let (size, input_size) = matrix_shape(w, "a token-wise layer's w", context)?;
    let rows = examples * tokens;
    if a.data.len() != rows * input_size {
        return Err(PyValueError::new_err(format!(
            "{context}: a token-wise layer of {tokens} tokens of {input_size} features got shape {:?}",
            a.shape
        )));
    }
    let out = forward(w, &reshaped(a, rows, input_size), b)?;
    Ok(reshaped(&out, examples, tokens * size))
}

fn reshaped(a: &RustArray, rows: usize, cols: usize) -> RustArray {
    RustArray {
        data: a.data.clone(),
        shape: Shape::Matrix(rows, cols),
    }
}

/// `window_program_forward`'s result: the outputs, and, as kept, each window's next state, its
/// cells' outputs before the snap, and its factors' code indices.
type ProgramForward = (RustArray, Vec<RustArray>, Vec<RustArray>, Vec<Vec<Option<Vec<usize>>>>);

/// The inference forward pass over every window of `x` `(N, W * k)` from the snapped initial state
/// `c0` `(N, S)`: each window's ops (Dropout passes its input on), then its recurrence step, as
/// indrajala-ml's `WindowRecurrentRustArrayNetwork.forward_windows` runs them. Returns `(outputs,
/// states, cells, indices)`: the outputs `(N, W * k * classes)`; with `states`, each window's next
/// state and its cells' outputs `(N, S)`, W of each; with `codes`, each window's code indices per
/// factor (None for a continuous factor). What isn't kept is an empty list.
#[pyfunction]
#[pyo3(signature = (program, x, parameters, c0, *, states = false, codes = false))]
pub fn window_program_forward(
    program: &WindowProgram,
    x: &RustArray,
    parameters: Vec<PyRef<'_, RustArray>>,
    c0: &RustArray,
    states: bool,
    codes: bool,
) -> PyResult<ProgramForward> {
    let context = "window_program_forward";
    let parameters: Vec<&RustArray> = parameters.iter().map(|p| &**p).collect();
    let split = program.split(&parameters, context)?;
    let (_, width) = matrix_shape(x, "x", context)?;
    let k = program.columns;
    if width == 0 || width % k != 0 {
        return Err(PyValueError::new_err(format!(
            "{context} reads whole windows of {k} columns, got {width} columns"
        )));
    }
    let total: usize = program.sizes.iter().sum();
    let (rows, _) = matrix_shape(x, "x", context)?;
    if c0.shape != Shape::Matrix(rows, total) {
        return Err(PyValueError::new_err(format!(
            "{context} requires c0 of shape ({rows}, {total}), got {:?}",
            c0.shape
        )));
    }
    let windows = width / k;
    let mut outputs = Vec::with_capacity(windows);
    let (mut kept_states, mut kept_cells, mut kept_indices) = (Vec::new(), Vec::new(), Vec::new());
    let mut c = c0.clone();
    for w in 0..windows {
        let (output, read) = program.window(columns(x, width, w * k, (w + 1) * k), &c, split.ops)?;
        let step = step_forward(
            &read,
            &c,
            &split.queries,
            &split.w,
            &split.u,
            &split.b,
            &split.codes,
            k,
            program.cross,
            context,
        )?;
        if step.sizes != program.sizes {
            return Err(PyValueError::new_err(format!(
                "{context}: the cells' state sizes {:?} aren't the program's {:?}",
                step.sizes, program.sizes
            )));
        }
        outputs.push(output);
        if codes {
            kept_indices.push(step.indices());
        }
        if states {
            kept_cells.push(step.h.clone());
            kept_states.push(step.state.clone());
        }
        c = step.state;
    }
    let refs: Vec<&RustArray> = outputs.iter().collect();
    Ok((join_columns(&refs, context)?, kept_states, kept_cells, kept_indices))
}

/// Each row's runs of `classes` values, each run's index of its largest, the first of equal
/// largest: Python's `token.index(max(token))` on the row's values, NaN as it (a value replaces
/// the largest so far only if it compares greater). `matrix` `(N, T * classes)`; one tuple a row.
#[pyfunction]
pub fn argmax_token_rows<'py>(
    py: Python<'py>,
    matrix: &RustArray,
    classes: usize,
) -> PyResult<Vec<Bound<'py, PyTuple>>> {
    let context = "argmax_token_rows";
    let (_, width) = matrix_shape(matrix, "matrix", context)?;
    if classes == 0 || width % classes != 0 {
        return Err(PyValueError::new_err(format!(
            "{context} requires rows of whole runs of classes >= 1 values, got {width} columns and classes {classes}"
        )));
    }
    matrix
        .data
        .chunks_exact(width.max(1))
        .map(|row| {
            let indices = row.chunks_exact(classes).map(|token| {
                let mut best = 0;
                for (i, &value) in token.iter().enumerate().skip(1) {
                    if value > token[best] {
                        best = i;
                    }
                }
                best
            });
            PyTuple::new(py, indices)
        })
        .collect()
}
