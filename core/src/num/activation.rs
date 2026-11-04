//! The hidden layers' activations (indrajala-ml's FFN activations workplan, D9): ReLU, GELU (the
//! exact, `erf` form; Hendrycks & Gimpel 2016), GELU-tanh (GPT-2's approximation) and SiLU
//! (`z * sigmoid(z)`, Elfwing et al. 2018; the gated FFN's, D3), one family the fused dense ops
//! ([`crate::nn::dense`]), the gated ops and the masks ([`crate::ops::mask::activation`]) take
//! as an argument.
//!
//! The expressions, their grouping fixed (indrajala-ml's pure-Python and numpy layers take the same
//! steps through their own `exp`, so the three agree by bits wherever `exp` does):
//!
//! ```text
//! gelu(z)       = (0.5 * z) * (1.0 + erf(z * SQRT1_2))
//! gelu'(z)      = 0.5 * (1.0 + erf(z * SQRT1_2)) + z * (exp(-0.5 * (z * z)) * INV_SQRT_2PI)
//! gelu_tanh(z)  = (0.5 * z) * (1.0 + th),  th = tanh(C * (z + K * ((z * z) * z)))
//! gelu_tanh'(z) = 0.5 * (1.0 + th) + (0.5 * z) * ((1.0 - th * th) * (C * (1.0 + (3.0 * K) * (z * z))))
//! silu(z)       = z * s,  s = 1.0 / (1.0 + exp(-z))
//! silu'(z)      = s * (1.0 + z * (1.0 - s))
//! ```
//!
//! `SQRT1_2` is the double nearest `1/sqrt(2)` (a multiply, not a division by `sqrt(2)`),
//! `INV_SQRT_2PI` the double nearest `1/sqrt(2 pi)`, `C` the double nearest `sqrt(2/pi)` and
//! `K = 0.044715`; `erf` is [`crate::num::erf::erf`] and `tanh` the window-recurrent cell's
//! `1 - 2 / (exp(2x) + 1)` (its W15), both through the correctly rounded [`crate::num::exp::exp`].
//! Only `+ - * /` besides, and Rust never contracts `a * b + c`, so the bits are the expressions',
//! the same on every machine.
//!
//! `s` is [`crate::num::exp::sigmoid_in_place`]'s, the dense layers' sigmoid.
//!
//! ReLU's derivative reads the layer's output (`a > 0`), the others' their input `z`: what the
//! forward keeps is [`Activation::keeps_preactivation`]'s answer, and a hidden delta's `cached`
//! operand is `a` for ReLU and `z` for the rest.
//!
//! The sigmoid is batch norm's (ReLU or sigmoid, [`crate::nn::norm::batch_norm`]), so the family
//! has one enum. [`Activation::from_name`] doesn't name it: the hidden layers' activation ops take
//! the FFN family only, and a dense layer's sigmoid is the plain [`crate::nn::dense`] ops'.

use crate::num::erf::erf;
use crate::num::exp::tanh_in_place;
use crate::num::exp::{exp, sigmoid_in_place};
use crate::num::scalar::relu;
use crate::Error;

/// The double nearest `1/sqrt(2)`.
pub const SQRT1_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// The double nearest `1/sqrt(2 pi)`: `0.3989422804014327`.
pub const INV_SQRT_2PI: f64 = f64::from_bits(0x3FD9_8845_33D4_3651);
/// The double nearest `sqrt(2/pi)`: `0.7978845608028654`.
pub const SQRT_2_OVER_PI: f64 = f64::from_bits(0x3FE9_8845_33D4_3651);
/// GPT-2's cubic coefficient.
pub const GELU_TANH_K: f64 = 0.044715;

/// A hidden layer's activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// `max(0, z)`, its derivative from the output: `a > 0`.
    Relu,
    /// `z * Phi(z)`, the exact form through `erf`.
    Gelu,
    /// GPT-2's `tanh` approximation of GELU.
    GeluTanh,
    /// `z * sigmoid(z)` (SiLU, or Swish with beta 1): SwiGLU's gate.
    Silu,
    /// `1 / (1 + exp(-z))`, the dense layers' sigmoid: batch norm's other activation. Its
    /// derivative `s * (1 - s)` is taken from `z` (the sigmoid recomputed), as the others' are.
    Sigmoid,
}

impl Activation {
    /// By the name indrajala-ml's specs give it: `"relu"`, `"gelu"`, `"gelu_tanh"` or `"silu"`.
    pub fn from_name(name: &str) -> Result<Self, Error> {
        match name {
            "relu" => Ok(Self::Relu),
            "gelu" => Ok(Self::Gelu),
            "gelu_tanh" => Ok(Self::GeluTanh),
            "silu" => Ok(Self::Silu),
            _ => Err(Error::invalid(format!(
                "unknown activation {name:?}: expected \"relu\", \"gelu\", \"gelu_tanh\" or \"silu\""
            ))),
        }
    }

    /// Whether the derivative reads the pre-activation `z` (all but ReLU), not the output `a`
    /// (ReLU): the forward returns `z` beside `a` exactly when this is true.
    pub fn keeps_preactivation(self) -> bool {
        !matches!(self, Self::Relu)
    }

    /// The activation of each of `z`.
    pub fn apply(self, z: &[f64]) -> Vec<f64> {
        match self {
            Self::Relu => z.iter().map(|&v| relu(v)).collect(),
            Self::Gelu => z.iter().map(|&v| gelu(v)).collect(),
            Self::GeluTanh => {
                let th = gelu_tanh_th(z);
                z.iter().zip(th.iter()).map(|(&v, &t)| (0.5 * v) * (1.0 + t)).collect()
            }
            Self::Silu => {
                let s = sigmoid_of(z);
                z.iter().zip(s.iter()).map(|(&v, &sv)| v * sv).collect()
            }
            Self::Sigmoid => sigmoid_of(z),
        }
    }

    /// [`Activation::apply`] in place, the bits the same: ReLU and the sigmoid without a copy
    /// (batch norm's), the others through `apply`.
    pub fn apply_in_place(self, values: &mut [f64]) {
        match self {
            Self::Relu => values.iter_mut().for_each(|v| *v = relu(*v)),
            Self::Sigmoid => sigmoid_in_place(values),
            Self::Gelu | Self::GeluTanh | Self::Silu => {
                let out = self.apply(values);
                values.copy_from_slice(&out);
            }
        }
    }

    /// `downstream` times the derivative, read from `cached` (`a` for ReLU, `z` for the rest).
    /// ReLU's is `d if a > 0 else 0`, the fused ops' mask as it always was. The two slices have
    /// the same length (the callers check the shapes).
    pub fn mask(self, downstream: &[f64], cached: &[f64]) -> Vec<f64> {
        match self {
            Self::Relu => downstream
                .iter()
                .zip(cached.iter())
                .map(|(&d, &av)| if av > 0.0 { d } else { 0.0 })
                .collect(),
            Self::Gelu => downstream
                .iter()
                .zip(cached.iter())
                .map(|(&d, &z)| d * gelu_derivative(z))
                .collect(),
            Self::GeluTanh => {
                let th = gelu_tanh_th(cached);
                downstream
                    .iter()
                    .zip(cached.iter().zip(th.iter()))
                    .map(|(&d, (&z, &t))| d * gelu_tanh_derivative_from(z, t))
                    .collect()
            }
            Self::Silu => {
                let s = sigmoid_of(cached);
                downstream
                    .iter()
                    .zip(cached.iter().zip(s.iter()))
                    .map(|(&d, (&z, &sv))| d * (sv * (1.0 + z * (1.0 - sv))))
                    .collect()
            }
            Self::Sigmoid => {
                let s = sigmoid_of(cached);
                downstream
                    .iter()
                    .zip(s.iter())
                    .map(|(&d, &sv)| d * sv * (1.0 - sv))
                    .collect()
            }
        }
    }
}

/// `gelu(z) = (0.5 * z) * (1.0 + erf(z * SQRT1_2))`.
#[inline]
pub fn gelu(z: f64) -> f64 {
    (0.5 * z) * (1.0 + erf(z * SQRT1_2))
}

/// `gelu'(z) = 0.5 * (1.0 + erf(z * SQRT1_2)) + z * (exp(-0.5 * (z * z)) * INV_SQRT_2PI)`.
#[inline]
pub fn gelu_derivative(z: f64) -> f64 {
    0.5 * (1.0 + erf(z * SQRT1_2)) + z * (exp(-0.5 * (z * z)) * INV_SQRT_2PI)
}

/// `th = tanh(C * (z + K * ((z * z) * z)))` for each of `z`, through the cell's `tanh_in_place`.
fn gelu_tanh_th(z: &[f64]) -> Vec<f64> {
    let mut th: Vec<f64> = z
        .iter()
        .map(|&v| SQRT_2_OVER_PI * (v + GELU_TANH_K * ((v * v) * v)))
        .collect();
    tanh_in_place(&mut th);
    th
}

/// `gelu_tanh'(z)` given its `th`.
#[inline]
fn gelu_tanh_derivative_from(z: f64, th: f64) -> f64 {
    0.5 * (1.0 + th) + (0.5 * z) * ((1.0 - th * th) * (SQRT_2_OVER_PI * (1.0 + (3.0 * GELU_TANH_K) * (z * z))))
}

/// `1 / (1 + exp(-z))` for each of `z`, the dense layers' sigmoid.
fn sigmoid_of(z: &[f64]) -> Vec<f64> {
    let mut s = z.to_vec();
    sigmoid_in_place(&mut s);
    s
}

/// `gelu_tanh(z)`, one value.
pub fn gelu_tanh(z: f64) -> f64 {
    Activation::GeluTanh.apply(&[z])[0]
}

/// `gelu_tanh'(z)`, one value.
pub fn gelu_tanh_derivative(z: f64) -> f64 {
    Activation::GeluTanh.mask(&[1.0], &[z])[0]
}
