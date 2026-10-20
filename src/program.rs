//! The window program (indrajala-ml's window-program workplan, stage 2): a window-recurrent
//! network's window, its layers as a list of ops, run over every window of an input in one call,
//! each window's layers then its factored recurrence step.
//!
//! ```text
//! window w      a_0 = x[:, w k : (w + 1) k],  a_(i+1) = op_i(a_i)   (the StateInputs read c_w)
//! step w        c_(w+1) = step(a_reads, c_w)                          (factored.rs's step_forward)
//!                (factor g's pooling reads the input of op reads[g]: one op for every factor, or
//!                each its own, the end of its branch)
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
use crate::attention::{attention_accumulate_gradient_batch, attention_downstream_batch, attention_forward_batch};
use crate::factored::{step_forward, sum_in_order, Consolidation, StepCore, StepGradients};
use crate::fused::{
    affine_forward_batch, layer_accumulate_gradient_batch, layer_downstream_batch, layer_relu_forward_batch,
    layer_softmax_forward_batch,
};
use crate::generator::Generator;
use crate::layer_norm::{layer_norm_accumulate_gradient_batch, layer_norm_downstream_batch, layer_norm_forward_batch};
use crate::recurrent::{columns, join_columns, matrix_shape, state_input_backward, state_input_forward};
use crate::tokens::{
    embedding_accumulate_gradient, embedding_forward, token_dropout_downstream, token_dropout_forward,
};
use crate::ufuncs::{array_relu_mask, sum_axis0};

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
    /// branches open (indrajala-ml's window branches workplan, D5): their input kept, each branch
    /// starting from it
    Split,
    /// a branch ends (its result kept for the poolings that read it) and the next starts from the
    /// innermost open split's input
    Branch,
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
    /// token-wise dropout, its drop probability: a mask drawn in training, nothing in inference
    Dropout { p: f64 },
}

impl Op {
    fn parameter_count(&self) -> usize {
        match self {
            Op::Embedding | Op::Position | Op::StateInput { .. } => 1,
            Op::Fork | Op::Add | Op::Split | Op::Branch | Op::Dropout { .. } => 0,
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
            "split" => {
                expect(0)?;
                Op::Split
            }
            "branch" => {
                expect(0)?;
                Op::Branch
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
                Op::Dropout { p: item(1)?.extract()? }
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
/// None for both), `("dropout", p)`, `("split",)`, `("branch",)`. `columns` is a window's width `k`
/// (its tokens), `reads` the op whose input the poolings read (one for every factor, or a list,
/// each factor's), `sizes` the factors' state sizes, `snapped` whether each has codes, `cross` the
/// cross-term.
///
/// Branches: `split` keeps its input; the ops after it are its first branch; each `branch` ends a
/// branch and starts the next from the split's input, the last branch running to the program's
/// end (the answers'). A factor reads its branch's result as the input of the `branch` op that ends
/// it. A residual block closes within its branch; a split stays open to the end.
#[pyclass(module = "indrajala_math_rust", frozen, skip_from_py_object)]
pub struct WindowProgram {
    ops: Vec<Op>,
    columns: usize,
    /// the op whose input each factor's pooling reads
    reads: Vec<usize>,
    sizes: Vec<usize>,
    snapped: Vec<bool>,
    cross: bool,
    /// each fork's add (the op that closes its block); 0 for any other op
    adds: Vec<usize>,
    /// each split's branch ops, in order; empty for any other op
    branches: Vec<Vec<usize>>,
}

/// `WindowProgram`'s `reads`: one op for every factor, or each factor's.
#[derive(FromPyObject)]
enum Reads {
    One(usize),
    Each(Vec<usize>),
}

/// A block open while the program's ops are checked.
enum Open {
    Fork(usize),
    Split(usize),
}

#[pymethods]
impl WindowProgram {
    #[new]
    #[pyo3(signature = (ops, columns, reads, sizes, snapped, *, cross))]
    fn new(
        ops: Vec<Bound<'_, PyTuple>>,
        columns: usize,
        reads: Reads,
        sizes: Vec<usize>,
        snapped: Vec<bool>,
        cross: bool,
    ) -> PyResult<Self> {
        let context = "WindowProgram";
        let ops = ops.iter().map(Op::parse).collect::<PyResult<Vec<_>>>()?;
        let reads = match reads {
            Reads::One(read) => vec![read; sizes.len()],
            Reads::Each(each) => each,
        };
        if ops.is_empty() || reads.len() != sizes.len() || reads.iter().any(|&r| r >= ops.len()) {
            return Err(PyValueError::new_err(format!(
                "{context} requires 1 or more ops and reads among them, one or one a factor, got {} ops and \
                 reads {reads:?} for {} factors",
                ops.len(),
                sizes.len()
            )));
        }
        if columns == 0 || sizes.is_empty() || sizes.len() != snapped.len() || sizes.contains(&0) {
            return Err(PyValueError::new_err(format!(
                "{context} requires columns >= 1 and one snapped flag per factor size, each size 1 or more, \
                 got columns {columns}, sizes {sizes:?}, snapped {snapped:?}"
            )));
        }
        let mut adds = vec![0; ops.len()];
        let mut branches = vec![Vec::new(); ops.len()];
        let mut open = Vec::new();
        for (i, op) in ops.iter().enumerate() {
            match op {
                Op::Fork => open.push(Open::Fork(i)),
                Op::Split => open.push(Open::Split(i)),
                Op::Add => {
                    let Some(Open::Fork(fork)) = open.pop() else {
                        return Err(PyValueError::new_err(format!(
                            "{context}: an add at op {i} without its fork in its branch"
                        )));
                    };
                    adds[fork] = i;
                }
                Op::Branch => {
                    let Some(Open::Split(split)) = open.last() else {
                        return Err(PyValueError::new_err(format!(
                            "{context}: a branch at op {i} without its split, or inside an open residual block"
                        )));
                    };
                    branches[*split].push(i);
                }
                _ => {}
            }
        }
        let forks_closed = open.iter().all(|block| matches!(block, Open::Split(_)));
        let splits_branched = open
            .iter()
            .all(|block| !matches!(block, Open::Split(split) if branches[*split].is_empty()));
        if !forks_closed || !splits_branched || !matches!(ops.last(), Some(Op::TokenOutput { .. })) {
            return Err(PyValueError::new_err(format!(
                "{context} requires every fork closed by an add, every split with a branch and a token_output op last"
            )));
        }
        Ok(WindowProgram {
            ops,
            columns,
            reads,
            sizes,
            snapped,
            cross,
            adds,
            branches,
        })
    }

    /// How many parameters a call takes: the ops', then the factors'.
    #[getter]
    fn parameter_count(&self) -> usize {
        self.op_parameters() + self.snapped.iter().map(|&s| 5 + usize::from(s)).sum::<usize>()
    }
}

/// A call's parameters (or the gradients beside them, in the same order) split by owner: the ops'
/// as one slice, and each factor's pooling query, cell `w`, `u`, `b` and codes.
struct Parameters<'a> {
    ops: &'a [&'a RustArray],
    queries: Vec<&'a RustArray>,
    w: Vec<&'a RustArray>,
    u: Vec<&'a RustArray>,
    b: Vec<&'a RustArray>,
    codes: Vec<Option<&'a RustArray>>,
}

/// Where each factor's arrays are in a call's list: its pooling query, then its cell's `w`, `u`
/// and `b`, its codes (if snapped) and its initial state.
struct FactorAt {
    query: usize,
    codes: Option<usize>,
}

/// What a window's op keeps for the backward pass.
enum Cache {
    None,
    LayerNorm {
        xhat: RustArray,
        std: RustArray,
    },
    Attention {
        q: RustArray,
        k: RustArray,
        v: RustArray,
        p: RustArray,
        h: RustArray,
        mask: Option<RustArray>,
    },
    Dropout {
        mask: RustArray,
    },
}

/// One window's forward pass, what its backward pass reads: each op's result (as the per-layer
/// path keeps its activations, the window's columns first), the activation each op read and the
/// one it gave (its input for a fork, or a dropout in inference), its cache, the tokens each
/// factor's pooling read, and the recurrence step.
struct WindowTape {
    activations: Vec<RustArray>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    caches: Vec<Cache>,
    read: Vec<usize>,
    step: StepCore,
}

impl WindowTape {
    fn floats(&self) -> usize {
        let caches: usize = self
            .caches
            .iter()
            .map(|cache| match cache {
                Cache::None => 0,
                Cache::LayerNorm { xhat, std } => xhat.data.len() + std.data.len(),
                Cache::Attention { q, k, v, p, h, mask } => {
                    q.data.len()
                        + k.data.len()
                        + v.data.len()
                        + p.data.len()
                        + h.data.len()
                        + mask.as_ref().map_or(0, |m| m.data.len())
                }
                Cache::Dropout { mask } => mask.data.len(),
            })
            .sum();
        self.activations.iter().map(|a| a.data.len()).sum::<usize>() + caches + self.step.floats()
    }

    /// The window's output, the last op's result.
    fn output(&self) -> &RustArray {
        &self.activations[*self.outputs.last().expect("a program has 1 or more ops")]
    }
}

impl WindowProgram {
    fn op_parameters(&self) -> usize {
        self.ops.iter().map(Op::parameter_count).sum()
    }

    /// Each op's first parameter's index in a call's list.
    fn op_offsets(&self) -> Vec<usize> {
        let mut at = 0;
        self.ops
            .iter()
            .map(|op| {
                let start = at;
                at += op.parameter_count();
                start
            })
            .collect()
    }

    fn factors_at(&self) -> Vec<FactorAt> {
        let mut at = self.op_parameters();
        self.snapped
            .iter()
            .map(|&snapped| {
                let factor = FactorAt {
                    query: at,
                    codes: snapped.then_some(at + 4),
                };
                at += 5 + usize::from(snapped); // the codes, then the initial state
                factor
            })
            .collect()
    }

    fn check_count(&self, count: usize, name: &str, context: &str) -> PyResult<()> {
        let expected = self.parameter_count();
        if count != expected {
            return Err(PyValueError::new_err(format!(
                "{context} requires {expected} {name}, the ops' then each factor's, got {count}"
            )));
        }
        Ok(())
    }

    fn split<'a>(&self, arrays: &'a [&'a RustArray]) -> Parameters<'a> {
        let factors = self.factors_at();
        Parameters {
            ops: &arrays[..self.op_parameters()],
            queries: factors.iter().map(|f| arrays[f.query]).collect(),
            w: factors.iter().map(|f| arrays[f.query + 1]).collect(),
            u: factors.iter().map(|f| arrays[f.query + 2]).collect(),
            b: factors.iter().map(|f| arrays[f.query + 3]).collect(),
            codes: factors.iter().map(|f| f.codes.map(|e| arrays[e])).collect(),
        }
    }

    /// One window's ops from its columns `x` and its state `c`, then its recurrence step: the
    /// window's tape. In training (`rng` given) dropout draws its masks from `rng` in the ops'
    /// order, as the per-layer path's layers draw them; in inference it passes its input on.
    /// Factor `g`'s codes are soft at `temperatures[g]` (None: hard); with `continuous[g]`, the
    /// factor passes its cell's output on unsnapped, its codes unread.
    #[allow(clippy::too_many_arguments)]
    fn window(
        &self,
        x: RustArray,
        c: &RustArray,
        split: &Parameters<'_>,
        rng: Option<&Bound<'_, Generator>>,
        temperatures: &[Option<f64>],
        continuous: &[bool],
        context: &str,
    ) -> PyResult<WindowTape> {
        let parameters = split.ops;
        let mut tape = WindowTape {
            activations: vec![x],
            inputs: Vec::with_capacity(self.ops.len()),
            outputs: Vec::with_capacity(self.ops.len()),
            caches: Vec::with_capacity(self.ops.len()),
            read: vec![0; self.reads.len()],
            step: StepCore::empty(),
        };
        let mut current = 0; // the activation the next op reads
        let mut forks: Vec<usize> = Vec::new();
        let mut splits: Vec<usize> = Vec::new();
        let mut at = 0;
        for (i, op) in self.ops.iter().enumerate() {
            for (read, &op_read) in tape.read.iter_mut().zip(&self.reads) {
                if op_read == i {
                    *read = current;
                }
            }
            let p = &parameters[at..at + op.parameter_count()];
            at += op.parameter_count();
            tape.inputs.push(current);
            let a = &tape.activations[current];
            let mut cache = Cache::None;
            let mut restart = None; // a branch's: the split's input, current again
            let next = match op {
                Op::Embedding => Some(embedding_forward(a, p[0])?),
                Op::Position => {
                    Some(a.combine_with_array(&RustArray::from_vector(p[0].data.clone()), |x, y| x + y, "add")?)
                }
                Op::StateInput { tokens } => Some(state_input_forward(a, c, p[0], *tokens)?),
                Op::Fork => {
                    forks.push(current);
                    None
                }
                Op::Split => {
                    splits.push(current);
                    None
                }
                Op::Branch => {
                    restart = splits.last().copied();
                    None
                }
                Op::Add => {
                    let Some(skip) = forks.pop() else {
                        return Err(PyValueError::new_err(format!("{context}: an add without its fork")));
                    };
                    Some(a.combine_with_array(&tape.activations[skip], |x, y| x + y, "add")?)
                }
                Op::LayerNorm { epsilon } => {
                    let (out, xhat, std) = layer_norm_forward_batch(a, p[0], p[1], *epsilon)?;
                    cache = Cache::LayerNorm { xhat, std };
                    Some(out)
                }
                Op::TokenDense { tokens, relu } => {
                    let forward = if *relu {
                        layer_relu_forward_batch
                    } else {
                        affine_forward_batch
                    };
                    Some(token_rows(a, *tokens, p[0], p[1], forward)?)
                }
                Op::TokenOutput { tokens } => Some(token_rows(a, *tokens, p[0], p[1], layer_softmax_forward_batch)?),
                Op::Attention {
                    heads,
                    causal,
                    dropout,
                    cos,
                    sin,
                } => {
                    let (out, q, k, v, pw, h, mask) = attention_forward_batch(
                        a,
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
                        rng.is_some(),
                        rng.map(|r| r.borrow_mut()),
                        cos.as_ref(),
                        sin.as_ref(),
                    )?;
                    if rng.is_some() {
                        cache = Cache::Attention {
                            q,
                            k,
                            v,
                            p: pw,
                            h,
                            mask,
                        };
                    }
                    Some(out)
                }
                Op::Dropout { p: drop } => match rng {
                    Some(rng) => {
                        let (out, mask) = token_dropout_forward(a, *drop, rng.borrow_mut())?;
                        cache = Cache::Dropout { mask };
                        Some(out)
                    }
                    None => None,
                },
            };
            if rng.is_none() {
                // inference keeps no cache: only the activations the later ops and the step read
                cache = Cache::None;
            }
            if let Some(next) = next {
                tape.activations.push(next);
                current = tape.activations.len() - 1;
            } else if let Some(split) = restart {
                current = split;
            }
            tape.outputs.push(current);
            tape.caches.push(cache);
        }
        let read: Vec<&RustArray> = tape.read.iter().map(|&r| &tape.activations[r]).collect();
        let codes: Vec<Option<&RustArray>> = split
            .codes
            .iter()
            .zip(continuous)
            .map(|(&e, &free)| if free { None } else { e })
            .collect();
        tape.step = step_forward(
            &read,
            c,
            &split.queries,
            &split.w,
            &split.u,
            &split.b,
            &codes,
            temperatures,
            self.columns,
            self.cross,
            context,
        )?;
        if tape.step.sizes != self.sizes {
            return Err(PyValueError::new_err(format!(
                "{context}: the cells' state sizes {:?} aren't the program's {:?}",
                tape.step.sizes, self.sizes
            )));
        }
        Ok(tape)
    }

    /// The factors run continuous in a call, one flag a factor (None: none), each a snapped
    /// factor at no temperature.
    fn check_continuous(
        &self,
        continuous: Option<Vec<bool>>,
        temperatures: &[Option<f64>],
        context: &str,
    ) -> PyResult<Vec<bool>> {
        let continuous = continuous.unwrap_or_else(|| vec![false; self.sizes.len()]);
        if continuous.len() != self.sizes.len() {
            return Err(PyValueError::new_err(format!(
                "{context} requires a continuous flag per factor, {} factors, got {}",
                self.sizes.len(),
                continuous.len()
            )));
        }
        for (g, ((&free, &snapped), temperature)) in continuous.iter().zip(&self.snapped).zip(temperatures).enumerate()
        {
            if free && (!snapped || temperature.is_some()) {
                return Err(PyValueError::new_err(format!(
                    "{context} runs only a snapped factor at no temperature continuous, factor {g} isn't"
                )));
            }
        }
        Ok(continuous)
    }

    fn check_input(&self, x: &RustArray, c0: &RustArray, context: &str) -> PyResult<usize> {
        let (rows, width) = matrix_shape(x, "x", context)?;
        let k = self.columns;
        if width == 0 || width % k != 0 {
            return Err(PyValueError::new_err(format!(
                "{context} reads whole windows of {k} columns, got {width} columns"
            )));
        }
        let total: usize = self.sizes.iter().sum();
        if c0.shape != Shape::Matrix(rows, total) {
            return Err(PyValueError::new_err(format!(
                "{context} requires c0 of shape ({rows}, {total}), got {:?}",
                c0.shape
            )));
        }
        Ok(width / k)
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
    let (_, input_size) = matrix_shape(w, "a token-wise layer's w", "window program")?;
    let (examples, rows) = token_shape(a, tokens, input_size)?;
    let out = forward(w, &reshaped(a, rows, input_size), b)?;
    let width = out.data.len() / examples.max(1);
    Ok(RustArray {
        data: out.data,
        shape: Shape::Matrix(examples, width),
    })
}

/// `a` `(N, T * features)`'s examples and its rows of tokens, `N * T`, refused unless whole.
fn token_shape(a: &RustArray, tokens: usize, features: usize) -> PyResult<(usize, usize)> {
    let context = "window program";
    let (examples, _) = matrix_shape(a, "a token-wise layer's array", context)?;
    let rows = examples * tokens;
    if a.data.len() != rows * features {
        return Err(PyValueError::new_err(format!(
            "{context}: a token-wise layer of {tokens} tokens of {features} features got shape {:?}",
            a.shape
        )));
    }
    Ok((examples, rows))
}

/// `a`'s values as a `(rows, cols)` matrix, a copy (as `Array.reshape`).
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
/// factor (None for a continuous factor). What isn't kept is an empty list. With `continuous`, a
/// flag a factor, a flagged factor passes its cell's output on unsnapped (indrajala-ml's state
/// formation workplan, the warm-up).
#[pyfunction]
#[pyo3(signature = (program, x, parameters, c0, *, states = false, codes = false, continuous = None))]
pub fn window_program_forward(
    program: &WindowProgram,
    x: &RustArray,
    parameters: Vec<PyRef<'_, RustArray>>,
    c0: &RustArray,
    states: bool,
    codes: bool,
    continuous: Option<Vec<bool>>,
) -> PyResult<ProgramForward> {
    let context = "window_program_forward";
    let parameters: Vec<&RustArray> = parameters.iter().map(|p| &**p).collect();
    program.check_count(parameters.len(), "parameters", context)?;
    let split = program.split(&parameters);
    let windows = program.check_input(x, c0, context)?;
    let (k, width) = (program.columns, windows * program.columns);
    let hard = vec![None; program.sizes.len()]; // inference reads the snapped machine
    let continuous = program.check_continuous(continuous, &hard, context)?;
    let mut outputs = Vec::with_capacity(windows);
    let (mut kept_states, mut kept_cells, mut kept_indices) = (Vec::new(), Vec::new(), Vec::new());
    let mut c = c0.clone();
    for w in 0..windows {
        let mut tape = program.window(
            columns(x, width, w * k, (w + 1) * k),
            &c,
            &split,
            None,
            &hard,
            &continuous,
            context,
        )?;
        let last = *tape.outputs.last().expect("a program has 1 or more ops");
        outputs.push(tape.activations.swap_remove(last));
        let step = tape.step;
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

/// One window's backward pass's gradients: each op's delta and what the ops before read of it.
struct Backward<'t> {
    program: &'t WindowProgram,
    tape: &'t WindowTape,
    parameters: &'t [&'t RustArray],
    offsets: &'t [usize],
    deltas: Vec<Option<RustArray>>,
    /// an attention's `(dx, dq, dk, dv)`, from its backward pass when its delta is set
    attention: Vec<Option<(RustArray, RustArray, RustArray, RustArray)>>,
    /// a StateInput's `(state delta, P's gradient)`
    state_inputs: Vec<Option<(RustArray, RustArray)>>,
}

impl Backward<'_> {
    fn delta(&self, j: usize) -> &RustArray {
        self.deltas[j]
            .as_ref()
            .expect("an op's delta is set before the ops before it read it")
    }

    /// Op `j`'s downstream, what the op before it reads (each indrajala-ml layer's
    /// `downstream_batch`); a fork's, its body's first op's plus its add's delta; a split's, its
    /// first branch's first op's plus each later branch's, in order. A branch's is never asked: what
    /// reads the result it ends is the poolings alone (the backward pass's loop).
    fn downstream(&self, j: usize) -> PyResult<RustArray> {
        let p = &self.parameters[self.offsets[j]..];
        let delta = || self.delta(j).clone();
        Ok(match &self.program.ops[j] {
            Op::Embedding => {
                return Err(PyValueError::new_err(
                    "window program: an embedding is the first op, nothing reads its downstream",
                ))
            }
            Op::Position | Op::Add | Op::StateInput { .. } => delta(),
            Op::Fork => {
                let add = self.program.adds[j];
                self.downstream(j + 1)?
                    .combine_with_array(self.delta(add), |x, y| x + y, "add")?
            }
            Op::Split => {
                let mut sum = self.downstream(j + 1)?;
                for &branch in &self.program.branches[j] {
                    sum = sum.combine_with_array(&self.downstream(branch + 1)?, |x, y| x + y, "add")?;
                }
                sum
            }
            Op::Branch => unreachable!("a branch's downstream: its result is read by the poolings alone"),
            Op::LayerNorm { .. } => {
                let Cache::LayerNorm { xhat, std } = &self.tape.caches[j] else {
                    unreachable!("a layer norm's cache")
                };
                layer_norm_downstream_batch(self.delta(j), p[0], xhat, std)?
            }
            Op::TokenDense { tokens, .. } | Op::TokenOutput { tokens } => {
                let (size, input_size) = matrix_shape(p[0], "a token-wise layer's w", "window program")?;
                let (examples, rows) = token_shape(self.delta(j), *tokens, size)?;
                let out = layer_downstream_batch(p[0], &reshaped(self.delta(j), rows, size))?;
                RustArray {
                    data: out.data,
                    shape: Shape::Matrix(examples, tokens * input_size),
                }
            }
            Op::Attention { .. } => self.attention[j]
                .as_ref()
                .expect("an attention's backward pass")
                .0
                .clone(),
            Op::Dropout { p: drop } => match &self.tape.caches[j] {
                Cache::Dropout { mask } => token_dropout_downstream(self.delta(j), mask, *drop)?,
                _ => delta(),
            },
        })
    }
}

/// The tape's step parameters' gradient view, each factor's, from the gradients list.
fn step_gradients<'a>(gradients: &'a [&'a RustArray], factors: &[FactorAt]) -> [Vec<&'a RustArray>; 4] {
    [
        factors.iter().map(|f| gradients[f.query]).collect(),
        factors.iter().map(|f| gradients[f.query + 1]).collect(),
        factors.iter().map(|f| gradients[f.query + 2]).collect(),
        factors.iter().map(|f| gradients[f.query + 3]).collect(),
    ]
}

impl WindowProgram {
    /// One window's backward pass after its forward pass (its tape), from the next state's
    /// gradient `dc_next` and its targets `y` `(N, k * classes)`, each duplicate code's state held
    /// to its representative's code by `consolidation`, as indrajala-ml's
    /// `WindowRecurrentShape._window_backward` runs it: each gradient accumulated onto its entry in
    /// `gradients`, and the window's state's gradient returned.
    #[allow(clippy::too_many_arguments)]
    fn window_backward(
        &self,
        tape: &WindowTape,
        c: &RustArray,
        y: &RustArray,
        dc_next: &RustArray,
        parameters: &[&RustArray],
        gradients: &mut [RustArray],
        weight: f64,
        commitments: &[f64],
        consolidation: &Consolidation,
        context: &str,
    ) -> PyResult<RustArray> {
        let split = self.split(parameters);
        let factors = self.factors_at();
        let targets = consolidation.targets(&tape.step, &split.codes);
        let targets: Vec<Option<&RustArray>> = targets.iter().map(Option::as_ref).collect();
        // the recurrence's step: its parts' gradients, and the poolings' gradient of the tokens
        let (pooled, dc_step, gq, gw, gu, gb, gcodes) = {
            let refs: Vec<&RustArray> = gradients.iter().collect();
            let [queries, w, u, b] = step_gradients(&refs, &factors);
            let snapped = tape.step.snapped();
            let codes: Vec<Option<&RustArray>> = factors
                .iter()
                .zip(snapped)
                .map(|(f, s)| if s { f.codes.map(|e| refs[e]) } else { None })
                .collect();
            let grads = StepGradients {
                queries: &queries,
                w: &w,
                u: &u,
                b: &b,
                codes: &codes,
            };
            let read: Vec<&RustArray> = tape.read.iter().map(|&r| &tape.activations[r]).collect();
            tape.step.backward(
                &read,
                dc_next,
                &split.queries,
                &split.w,
                &split.u,
                &grads,
                weight,
                commitments,
                &targets,
                context,
            )?
        };
        for (f, (q, (w, (u, (b, e))))) in factors.iter().zip(
            gq.into_iter()
                .zip(gw.into_iter().zip(gu.into_iter().zip(gb.into_iter().zip(gcodes)))),
        ) {
            gradients[f.query] = q;
            gradients[f.query + 1] = w;
            gradients[f.query + 2] = u;
            gradients[f.query + 3] = b;
            if let (Some(at), Some(e)) = (f.codes, e) {
                gradients[at] = e;
            }
        }

        let offsets = self.op_offsets();
        let last = self.ops.len() - 1;
        let mut back = Backward {
            program: self,
            tape,
            parameters,
            offsets: &offsets,
            deltas: vec![None; self.ops.len()],
            attention: (0..self.ops.len()).map(|_| None).collect(),
            state_inputs: (0..self.ops.len()).map(|_| None).collect(),
        };
        // the output layer's delta, (P - Y) / T, then times 1 / W
        let Op::TokenOutput { tokens } = self.ops[last] else {
            return Err(PyValueError::new_err(format!(
                "{context}: a window ends in its token_output op"
            )));
        };
        let a = tape.output();
        let difference = a.combine_with_array(y, |p, t| p - t, "subtract")?;
        let delta = scaled(&scaled(&difference, |v| v / tokens as f64), |v| v * weight);
        back.deltas[last] = Some(delta);
        // the hidden ops' deltas, last first; each factor's pooling's gradient joins the downstream
        // the op before the one it reads reads (those reading one op summed in factor order first);
        // a branch's end is read by its poolings alone
        for i in (0..last).rev() {
            if matches!(self.ops[i], Op::Fork | Op::Split | Op::Branch) {
                // its delta is never read: a fork's downstream is its body's and its add's, a
                // split's its branches', and a branch has none
                continue;
            }
            let readers: Vec<usize> = (0..self.reads.len()).filter(|&g| self.reads[g] == i + 1).collect();
            let pooling = sum_in_order(&pooled, &readers);
            let mut d = if matches!(self.ops[i + 1], Op::Branch) {
                pooling.ok_or_else(|| {
                    PyValueError::new_err(format!("{context}: the branch at op {} ends unread", i + 1))
                })?
            } else {
                let d = back.downstream(i + 1)?;
                match pooling {
                    Some(pooling) => d.combine_with_array(&pooling, |x, y| x + y, "add")?,
                    None => d,
                }
            };
            let p = &parameters[offsets[i]..];
            match &self.ops[i] {
                Op::TokenDense { relu: true, .. } => {
                    d = array_relu_mask(&d, &tape.activations[tape.outputs[i]])?;
                }
                Op::Attention {
                    heads,
                    dropout,
                    cos,
                    sin,
                    ..
                } => {
                    let Cache::Attention {
                        q, k, v, p: pw, mask, ..
                    } = &tape.caches[i]
                    else {
                        unreachable!("an attention's cache in training")
                    };
                    back.attention[i] = Some(attention_downstream_batch(
                        &d,
                        p[0],
                        p[2],
                        p[4],
                        p[6],
                        q,
                        k,
                        v,
                        pw,
                        *heads,
                        mask.as_ref(),
                        *dropout,
                        cos.as_ref(),
                        sin.as_ref(),
                    )?);
                }
                Op::StateInput { tokens } => {
                    let grad_p = &gradients[offsets[i]];
                    back.state_inputs[i] = Some(state_input_backward(&d, c, p[0], grad_p, *tokens)?);
                }
                _ => {}
            }
            back.deltas[i] = Some(d);
        }
        // each op's gradients accumulated, from its delta and the activation it read
        for (i, op) in self.ops.iter().enumerate() {
            let at = offsets[i];
            let input = &tape.activations[tape.inputs[i]];
            match op {
                Op::Embedding => {
                    gradients[at] = embedding_accumulate_gradient(back.delta(i), input, &gradients[at])?;
                }
                Op::Position => {
                    let sum = sum_axis0(back.delta(i))?;
                    let shaped = RustArray {
                        data: sum.data,
                        shape: gradients[at].shape,
                    };
                    gradients[at] = gradients[at].combine_with_array(&shaped, |x, y| x + y, "add")?;
                }
                Op::StateInput { .. } => {
                    gradients[at] = back.state_inputs[i]
                        .as_ref()
                        .expect("a StateInput's backward pass")
                        .1
                        .clone();
                }
                Op::LayerNorm { .. } => {
                    let Cache::LayerNorm { xhat, .. } = &tape.caches[i] else {
                        unreachable!("a layer norm's cache")
                    };
                    let (gg, gbeta) =
                        layer_norm_accumulate_gradient_batch(back.delta(i), xhat, &gradients[at], &gradients[at + 1])?;
                    gradients[at] = gg;
                    gradients[at + 1] = gbeta;
                }
                Op::TokenDense { tokens, .. } | Op::TokenOutput { tokens } => {
                    let (size, input_size) = matrix_shape(parameters[at], "a token-wise layer's w", context)?;
                    let (_, rows) = token_shape(back.delta(i), *tokens, size)?;
                    let (gw, gbias) = layer_accumulate_gradient_batch(
                        &reshaped(back.delta(i), rows, size),
                        &reshaped(input, rows, input_size),
                        &gradients[at],
                        &gradients[at + 1],
                    )?;
                    gradients[at] = gw;
                    gradients[at + 1] = gbias;
                }
                Op::Attention { .. } => {
                    let Cache::Attention { h, .. } = &tape.caches[i] else {
                        unreachable!("an attention's cache in training")
                    };
                    let (_, dq, dk, dv) = back.attention[i].as_ref().expect("an attention's backward pass");
                    let g = &gradients[at..at + 8];
                    let new = attention_accumulate_gradient_batch(
                        back.delta(i),
                        input,
                        h,
                        dq,
                        dk,
                        dv,
                        &g[0],
                        &g[1],
                        &g[2],
                        &g[3],
                        &g[4],
                        &g[5],
                        &g[6],
                        &g[7],
                    )?;
                    let (a0, a1, a2, a3, a4, a5, a6, a7) = new;
                    for (slot, value) in gradients[at..at + 8].iter_mut().zip([a0, a1, a2, a3, a4, a5, a6, a7]) {
                        *slot = value;
                    }
                }
                Op::Fork | Op::Add | Op::Split | Op::Branch | Op::Dropout { .. } => {}
            }
        }
        // the state's gradient: the step's, then each StateInput's, in the ops' order
        let mut dc = dc_step;
        for state_input in back.state_inputs.iter().flatten() {
            dc = dc.combine_with_array(&state_input.0, |x, y| x + y, "add")?;
        }
        Ok(dc)
    }
}

/// `a` with `f` applied to each value, as an `Array`'s scalar operation.
fn scaled(a: &RustArray, f: impl Fn(f64) -> f64) -> RustArray {
    RustArray {
        data: a.data.iter().map(|&v| f(v)).collect(),
        shape: a.shape,
    }
}

/// The default bound on what a training call keeps of its windows' forward passes (D5): 2 GiB.
const MEMORY_BUDGET: usize = 2 << 30;

/// A training step's gradients over every window of `x` `(N, W * k)` with targets `y`
/// `(N, W * k * classes)`, from the snapped initial state `c0`, as indrajala-ml's
/// `WindowRecurrentShape._accumulate_gradients` computes them: the windows forward in training,
/// each op's dropout masks drawn from `rng` in the per-layer path's order, keeping every window's
/// results (D5: no recomputed forward pass); then backward from the last window, each window's
/// gradients accumulated onto `gradients` (each parameter's, in `parameters`' order; each entry
/// replaced by its sum) with the codes' losses `weight` (`1 / W`) times `commitments`. With a
/// `consolidation`, pairs of joint codes (a code index per factor) `(duplicate, representative)`,
/// a window's state snapped to a duplicate is held by the commitment to its representative's codes
/// instead (indrajala-ml's code-consolidation workplan, D1); empty, as without it. Returns the
/// initial state's gradient `(N, S)`, which the caller takes through the initial state's codes.
/// With `temperatures`, factor `g`'s codes are soft at `temperatures[g]` in every window (None, or
/// no `temperatures`, hard; indrajala-ml's commitment-control workplan, stage 3). With
/// `continuous`, a flag a factor, a flagged factor passes its cell's output on unsnapped and its
/// codes' gradients are left as they are (indrajala-ml's state formation workplan, the warm-up).
/// Refused before the windows past the first run when what they keep would pass `memory_budget`
/// bytes.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (program, x, y, parameters, gradients, c0, rng, weight, commitments, *, consolidation = Vec::new(), memory_budget = MEMORY_BUDGET, temperatures = None, continuous = None))]
pub fn window_program_learn(
    program: &WindowProgram,
    x: &RustArray,
    y: &RustArray,
    parameters: Vec<PyRef<'_, RustArray>>,
    mut gradients: Vec<PyRefMut<'_, RustArray>>,
    c0: &RustArray,
    rng: Bound<'_, Generator>,
    weight: f64,
    commitments: Vec<f64>,
    consolidation: Vec<(Vec<usize>, Vec<usize>)>,
    memory_budget: usize,
    temperatures: Option<Vec<Option<f64>>>,
    continuous: Option<Vec<bool>>,
) -> PyResult<RustArray> {
    let context = "window_program_learn";
    let temperatures = temperatures.unwrap_or_else(|| vec![None; program.sizes.len()]);
    if temperatures.len() != program.sizes.len() {
        return Err(PyValueError::new_err(format!(
            "{context} requires a temperature per factor, {} factors, got {}",
            program.sizes.len(),
            temperatures.len()
        )));
    }
    let continuous = program.check_continuous(continuous, &temperatures, context)?;
    if continuous.contains(&true) && !consolidation.is_empty() {
        return Err(PyValueError::new_err(format!(
            "{context} takes no consolidation with a continuous factor"
        )));
    }
    let parameters: Vec<&RustArray> = parameters.iter().map(|p| &**p).collect();
    program.check_count(parameters.len(), "parameters", context)?;
    program.check_count(gradients.len(), "gradients", context)?;
    if commitments.len() != program.sizes.len() {
        return Err(PyValueError::new_err(format!(
            "{context} requires a commitment per factor, {} factors, got {}",
            program.sizes.len(),
            commitments.len()
        )));
    }
    let windows = program.check_input(x, c0, context)?;
    let (k, width) = (program.columns, windows * program.columns);
    let (rows, _) = matrix_shape(x, "x", context)?;
    let (y_rows, targets) = matrix_shape(y, "y", context)?;
    if y_rows != rows || targets % windows != 0 {
        return Err(PyValueError::new_err(format!(
            "{context} requires y of one row an example and whole windows of targets, got {:?}",
            y.shape
        )));
    }
    let per_window = targets / windows;
    let split = program.split(&parameters);
    let consolidation = Consolidation::new(consolidation, &split.codes, context)?;

    let mut tapes = Vec::with_capacity(windows);
    let mut states = Vec::with_capacity(windows);
    let mut c = c0.clone();
    for w in 0..windows {
        let tape = program.window(
            columns(x, width, w * k, (w + 1) * k),
            &c,
            &split,
            Some(&rng),
            &temperatures,
            &continuous,
            context,
        )?;
        if w == 0 {
            let bytes = tape.floats() * std::mem::size_of::<f64>() * windows;
            if bytes > memory_budget {
                return Err(PyValueError::new_err(format!(
                    "{context}: keeping {windows} windows' forward passes takes about {bytes} bytes, past the \
                     memory budget of {memory_budget} (memory_budget)"
                )));
            }
        }
        let next = tape.step.state.clone();
        states.push(std::mem::replace(&mut c, next));
        tapes.push(tape);
    }

    // every gradient taken out of its array for the pass and put back, whatever its result
    let mut owned: Vec<RustArray> = gradients
        .iter_mut()
        .map(|g| std::mem::replace(&mut **g, RustArray::from_vector(Vec::new())))
        .collect();
    let total: usize = program.sizes.iter().sum();
    let mut dc = RustArray::from_matrix(vec![0.0; rows * total], rows, total);
    let mut outcome = Ok(());
    for w in (0..windows).rev() {
        let yw = columns(y, targets, w * per_window, (w + 1) * per_window);
        match program.window_backward(
            &tapes[w],
            &states[w],
            &yw,
            &dc,
            &parameters,
            &mut owned,
            weight,
            &commitments,
            &consolidation,
            context,
        ) {
            Ok(next) => dc = next,
            Err(error) => {
                outcome = Err(error);
                break;
            }
        }
    }
    for (g, value) in gradients.iter_mut().zip(owned) {
        **g = value;
    }
    outcome.map(|()| dc)
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
