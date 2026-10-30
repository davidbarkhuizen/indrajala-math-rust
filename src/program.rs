//! The bindings of the core's window program (`indrajala_math::program`): ops read from their
//! Python tuples, list arguments lent to the core, and the generator borrowed for a learning call.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::array::RustArray;
use crate::generator::Generator;
use crate::linalg::py_err;
use indrajala_math::program::{Op, ProgramForward, Reads, MEMORY_BUDGET};
use indrajala_math::Array;

/// The op a descriptor names: `(kind, *settings)`, as `WindowProgram`'s documentation lists.
fn parse_op(descriptor: &Bound<'_, PyTuple>) -> PyResult<Op> {
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
        "merge" => {
            expect(0)?;
            Op::Merge
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
            let table = |i: usize| -> PyResult<Option<Array>> {
                let value = item(i)?;
                if value.is_none() {
                    Ok(None)
                } else {
                    Ok(Some(value.extract::<PyRef<'_, RustArray>>()?.0.clone()))
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
                "{context} has no op {kind:?}: embedding, position, state_input, fork, add, split, branch, \
                 merge, layer_norm, token_dense, token_output, attention, dropout"
            )))
        }
    })
}

/// A window-recurrent network's window as ops, and its recurrence's shape: what
/// `window_program_forward` runs.
///
/// `ops` are tuples, a kind then its settings: `("embedding",)`, `("position",)`,
/// `("state_input", tokens)`, `("fork",)`, `("add",)`, `("layer_norm", epsilon)`,
/// `("token_dense", tokens, "relu" | "linear")`, `("token_output", tokens)`,
/// `("attention", heads, causal, dropout, cos, sin)` (`cos`, `sin` the rotary table's arrays, or
/// None for both), `("dropout", p)`, `("split",)`, `("branch",)`, `("merge",)`. `columns` is a window's width `k`
/// (its tokens), `reads` the op whose input the poolings read (one for every factor, or a list,
/// each factor's), `sizes` the factors' state sizes, `snapped` whether each has codes, `cross` the
/// cross-term.
///
/// Branches: `split` keeps its input; the ops after it are its first branch; each `branch` ends a
/// branch and starts the next from the split's input, the last branch running to the program's
/// end (the answers'), or to a `merge`. A factor reads its branch's result as the input of the
/// `branch` op that ends it (the last branch's, the `merge`'s input). A `merge` closes the innermost
/// open split, its result the sum of its branches' ends in order, and the ops after it read that
/// sum (indrajala-ml's post-paper workplan, D7: the recombining window). A residual block closes
/// within its branch; a split stays open to the end unless merged.
#[pyclass(module = "indrajala_math_rust", frozen, skip_from_py_object)]
pub struct WindowProgram(pub indrajala_math::program::WindowProgram);

/// `WindowProgram`'s `reads`: one op for every factor, or each factor's.
#[derive(FromPyObject)]
enum PyReads {
    One(usize),
    Each(Vec<usize>),
}

#[pymethods]
impl WindowProgram {
    #[new]
    #[pyo3(signature = (ops, columns, reads, sizes, snapped, *, cross))]
    fn new(
        ops: Vec<Bound<'_, PyTuple>>,
        columns: usize,
        reads: PyReads,
        sizes: Vec<usize>,
        snapped: Vec<bool>,
        cross: bool,
    ) -> PyResult<Self> {
        let ops = ops.iter().map(parse_op).collect::<PyResult<Vec<_>>>()?;
        let reads = match reads {
            PyReads::One(read) => Reads::One(read),
            PyReads::Each(each) => Reads::Each(each),
        };
        indrajala_math::program::WindowProgram::new(ops, columns, reads, sizes, snapped, cross)
            .map(WindowProgram)
            .map_err(py_err)
    }

    /// How many parameters a call takes: the ops', then the factors'.
    #[getter]
    fn parameter_count(&self) -> usize {
        self.0.parameter_count()
    }
}

/// `ProgramForward` with the bindings' arrays.
type PyProgramForward = (RustArray, Vec<RustArray>, Vec<RustArray>, Vec<Vec<Option<Vec<usize>>>>);

/// The inference forward pass over every window of `x` `(N, W * k)` from the snapped initial state
/// `c0` `(N, S)`: each window's ops (Dropout passes its input on), then its recurrence step, as
/// indrajala-ml's `WindowRecurrentRustArrayNetwork.forward_windows` runs them. Returns `(outputs,
/// states, cells, indices)`: the outputs `(N, W * k * classes)`; with `states`, each window's next
/// state and its cells' outputs `(N, S)`, W of each; with `codes`, each window's code indices per
/// factor (None for a continuous factor). What isn't kept is an empty list. With `continuous`, a
/// flag a factor, a flagged factor passes its cell's output on unsnapped (indrajala-ml's state
/// formation workplan, the warm-up); with `couplings`, factor `g`'s next state `(1 - couplings[g])
/// h + couplings[g] q` (the workplan's gradual handover; None, every 1).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (program, x, parameters, c0, *, states = false, codes = false, continuous = None, couplings = None))]
pub fn window_program_forward(
    program: &WindowProgram,
    x: &RustArray,
    parameters: Vec<PyRef<'_, RustArray>>,
    c0: &RustArray,
    states: bool,
    codes: bool,
    continuous: Option<Vec<bool>>,
    couplings: Option<Vec<f64>>,
) -> PyResult<PyProgramForward> {
    let parameters: Vec<&Array> = parameters.iter().map(|p| &p.0).collect();
    let arrays = |v: Vec<Array>| v.into_iter().map(RustArray::from).collect::<Vec<_>>();
    indrajala_math::program::window_program_forward(
        &program.0,
        x,
        &parameters,
        c0,
        states,
        codes,
        continuous,
        couplings,
    )
    .map(|(outputs, kept_states, kept_cells, kept_indices): ProgramForward| {
        (outputs.into(), arrays(kept_states), arrays(kept_cells), kept_indices)
    })
    .map_err(py_err)
}

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
/// With `couplings`, factor `g`'s next state `(1 - couplings[g]) h + couplings[g] q`, its snapped
/// path's gradient `couplings[g] dq` (its codes' losses whole), the cell's `(1 - couplings[g]) dq`
/// (the workplan's gradual handover; None, every 1). With `straight`, a soft factor's gradient is
/// straight through (`soft_codes_backward`'s; the workplan's D51). Refused before the windows past the first run when what they keep would pass `memory_budget`
/// bytes.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (program, x, y, parameters, gradients, c0, rng, weight, commitments, *, consolidation = Vec::new(), memory_budget = MEMORY_BUDGET, temperatures = None, continuous = None, couplings = None, straight = false))]
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
    couplings: Option<Vec<f64>>,
    straight: bool,
) -> PyResult<RustArray> {
    let parameters: Vec<&Array> = parameters.iter().map(|p| &p.0).collect();
    let mut gradients: Vec<&mut Array> = gradients.iter_mut().map(|g| &mut g.0).collect();
    indrajala_math::program::window_program_learn(
        &program.0,
        x,
        y,
        &parameters,
        &mut gradients,
        c0,
        &mut rng.borrow_mut().0,
        weight,
        commitments,
        consolidation,
        memory_budget,
        temperatures,
        continuous,
        couplings,
        straight,
    )
    .map(RustArray::from)
    .map_err(py_err)
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
    indrajala_math::program::argmax_token_rows(matrix, classes)
        .map_err(py_err)?
        .into_iter()
        .map(|row| PyTuple::new(py, row))
        .collect()
}
