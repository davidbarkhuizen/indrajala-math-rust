//! Correctly rounded binary64 `exp` (the aarch64 workplan's D4, in indrajala-ml): `exp(x)` is the
//! binary64 value nearest the exact `e^x`, ties to even, so its bits are defined by mathematics,
//! not by a C library or a compiler, and every machine, and any later vector `exp`, gives the same.
//!
//! A port of CORE-MATH's `cr_exp` (src/binary64/exp/exp.c at core-math dae5dd8c, Alexei
//! Sibidanov), whose algorithm and proof it follows step for step: a fast path in double precision
//! with a rounding test (`ub == lb`), and, when the test fails (rarely), an accurate path in
//! double-double with a database of the inputs it alone can't round. Tables generated from the C
//! source's hexadecimal literals (each beside its bits). What the port leaves out: errno and the
//! floating-point exception flags (nothing in this crate reads them). Rust never contracts
//! `a * b + c` on its own, which is the C source's "without FMA contraction" build; `mul_add`
//! is the C source's `__builtin_fma`, correctly rounded with or without the hardware's FMA.
//!
//! The original's licence:
//!
//! Copyright (c) 2022-2026 Alexei Sibidanov <sibid@uvic.ca>.
//!
//! This file is part of the CORE-MATH project (<https://core-math.gitlabpages.inria.fr/>).
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy of this software
//! and associated documentation files (the "Software"), to deal in the Software without
//! restriction, including without limitation the rights to use, copy, modify, merge, publish,
//! distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
//! Software is furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all copies or
//! substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
//! BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//! NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//! DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/// `e^x`, correctly rounded. On x86_64 the body runs with the hardware's FMA when the CPU has it
/// (the same bits either way: `mul_add` is correctly rounded on both paths), as the crate's
/// matmul kernels do.
#[inline]
pub fn exp(x: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("fma") {
            // SAFETY: the CPU has FMA, checked just above.
            return unsafe { exp_fma(x) };
        }
    }
    exp_body(x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn exp_fma(x: f64) -> f64 {
    exp_body(x)
}

/// `exp` of each of `values` into `out` (the same length), in vector lanes where the CPU has them:
/// AVX2 and FMA's 4 on x86_64 (checked once per call), NEON's 2 on aarch64 (indrajala-ml's crate
/// kernels workplan, D1 and D2). Each lane runs [`exp`]'s fast path, operation for operation, so a
/// lane that rounds gives `exp`'s bits; a lane whose rounding test fails, or whose input is outside
/// the fast path's ordinary range (`|x| <= 2^-54`, a subnormal or infinite result, NaN), is redone
/// by `exp` itself. Returns how many values were redone.
pub fn exp_into(values: &[f64], out: &mut [f64]) -> usize {
    assert_eq!(values.len(), out.len(), "exp_into: values and out differ in length");
    // SAFETY: both hold values.len() values, and a borrow and a mutable borrow can't overlap.
    unsafe { exp_lanes(values.as_ptr(), out.as_mut_ptr(), values.len()) }
}

/// [`exp_into`] with `values` as its own `out`: each vector is loaded before its results are
/// stored, and a redone lane reads its input from the loaded vector.
pub fn exp_in_place(values: &mut [f64]) -> usize {
    let data = values.as_mut_ptr();
    // SAFETY: `data` holds values.len() values; the kernels read each vector before writing it.
    unsafe { exp_lanes(data, data, values.len()) }
}

// The kernels read `len` values at `src` and write `len` at `dst`, which is `src` itself or doesn't
// overlap it. Safety: both hold `len` values.
unsafe fn exp_lanes(src: *const f64, dst: *mut f64, len: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: the CPU has AVX2 and FMA, checked just above.
            return unsafe { exp_lanes_avx2(src, dst, len) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of aarch64.
        return unsafe { exp_lanes_neon(src, dst, len) };
    }
    #[allow(unreachable_code, reason = "aarch64 returns above")]
    {
        for i in 0..len {
            *dst.add(i) = exp(*src.add(i));
        }
        len
    }
}

// The vector lanes' ordinary inputs, as `|x|`'s bits: above 2^-54 (smaller rounds to 1 + x) and
// below the subnormal results' edge, which also keeps out the overflows, the infinities and NaN.
const ORDINARY_ABOVE: u64 = 0x3c90000000000000;
const ORDINARY_BELOW: u64 = SUBNORMAL_BELOW & (u64::MAX >> 1);
// 0x1.8p+52: an integer `t` with `|t| < 2^51` added to it lands in the mantissa, `t + 2^51`.
#[cfg(target_arch = "x86_64")]
const ROUND_MAGIC: f64 = 6755399441055744.0;

// `exp_body`'s fast path in AVX2, four lanes at a time; the tables' pairs `[low, high]` gathered by
// index. Safety: the CPU has AVX2 and FMA; as `exp_lanes`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_lanes_avx2(src: *const f64, dst: *mut f64, len: usize) -> usize {
    use std::arch::x86_64::*;

    let pd = _mm256_set1_pd;
    let epi = |bits: u64| _mm256_set1_epi64x(bits as i64);
    let (t0, t1) = (T0.as_ptr().cast::<f64>(), T1.as_ptr().cast::<f64>());
    let quads = len & !3;
    let mut redone = 0;
    for i in (0..quads).step_by(4) {
        let x = _mm256_loadu_pd(src.add(i));
        let abs_bits = _mm256_and_si256(_mm256_castpd_si256(x), epi(u64::MAX >> 1));
        // signed compares, exact here: |x|'s bits have the sign bit clear
        let ordinary = _mm256_and_si256(
            _mm256_cmpgt_epi64(abs_bits, epi(ORDINARY_ABOVE)),
            _mm256_cmpgt_epi64(epi(ORDINARY_BELOW), abs_bits),
        );
        // reduce: t, jt and the table product (th, tl); an extraordinary lane's indices are
        // garbage but masked to the tables' 64 rows, and its result is replaced
        let t = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_pd(x, pd(S)));
        let jt = _mm256_sub_epi64(
            _mm256_castpd_si256(_mm256_add_pd(t, pd(ROUND_MAGIC))),
            _mm256_castpd_si256(pd(ROUND_MAGIC)),
        );
        let i1 = _mm256_slli_epi64::<1>(_mm256_and_si256(jt, epi(0x3f)));
        let i0 = _mm256_slli_epi64::<1>(_mm256_and_si256(_mm256_srli_epi64::<6>(jt), epi(0x3f)));
        let one = epi(1);
        let (xl, xh) = (
            _mm256_i64gather_pd::<8>(t0, i0),
            _mm256_i64gather_pd::<8>(t0, _mm256_add_epi64(i0, one)),
        );
        let (cl, ch) = (
            _mm256_i64gather_pd::<8>(t1, i1),
            _mm256_i64gather_pd::<8>(t1, _mm256_add_epi64(i1, one)),
        );
        let th = _mm256_mul_pd(ch, xh);
        let tl = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(ch, xl), _mm256_mul_pd(cl, xh)),
            _mm256_fmsub_pd(ch, xh, th),
        );
        // the fast path, as exp_body has it
        let dx = _mm256_add_pd(_mm256_sub_pd(x, _mm256_mul_pd(pd(L2H), t)), _mm256_mul_pd(pd(L2L), t));
        let dx2 = _mm256_mul_pd(dx, dx);
        let p = _mm256_add_pd(
            _mm256_add_pd(pd(FAST_C[0]), _mm256_mul_pd(dx, pd(FAST_C[1]))),
            _mm256_mul_pd(dx2, _mm256_add_pd(pd(FAST_C[2]), _mm256_mul_pd(dx, pd(FAST_C[3])))),
        );
        let tx = _mm256_mul_pd(th, dx);
        let fl = _mm256_add_pd(tl, _mm256_mul_pd(tx, p));
        let ub = _mm256_add_pd(th, _mm256_add_pd(fl, pd(EPS)));
        let lb = _mm256_add_pd(th, _mm256_sub_pd(fl, pd(EPS)));
        // ldexp_bits(lb, ie): ie << 52 is (jt >> 12) << 52, the same bits as (jt & !0xfff) << 40
        let scale = _mm256_slli_epi64::<40>(_mm256_and_si256(jt, epi(!0xfff)));
        let result = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(lb), scale));
        _mm256_storeu_pd(dst.add(i), result);
        let rounded = _mm256_and_pd(_mm256_cmp_pd::<_CMP_EQ_OQ>(ub, lb), _mm256_castsi256_pd(ordinary));
        let mask = _mm256_movemask_pd(rounded);
        if mask != 0b1111 {
            let mut xs = [0.0; 4];
            _mm256_storeu_pd(xs.as_mut_ptr(), x);
            for (lane, &v) in xs.iter().enumerate() {
                if mask & (1 << lane) == 0 {
                    *dst.add(i + lane) = exp_body(v);
                    redone += 1;
                }
            }
        }
    }
    for i in quads..len {
        *dst.add(i) = exp_body(*src.add(i));
    }
    redone + (len - quads)
}

// `exp_body`'s fast path in NEON, two lanes at a time; the table rows loaded per lane. Safety: as
// `exp_lanes`.
#[cfg(target_arch = "aarch64")]
unsafe fn exp_lanes_neon(src: *const f64, dst: *mut f64, len: usize) -> usize {
    use std::arch::aarch64::*;

    let pd = vdupq_n_f64;
    let pairs = len & !1;
    let mut redone = 0;
    for i in (0..pairs).step_by(2) {
        let x = vld1q_f64(src.add(i));
        let abs_bits = vandq_u64(vreinterpretq_u64_f64(x), vdupq_n_u64(u64::MAX >> 1));
        let ordinary = vandq_u64(
            vcgtq_u64(abs_bits, vdupq_n_u64(ORDINARY_ABOVE)),
            vcltq_u64(abs_bits, vdupq_n_u64(ORDINARY_BELOW)),
        );
        // reduce: t, jt and the table product (th, tl); an extraordinary lane's jt saturates or is
        // garbage, its indices masked to the tables' 64 rows, and its result is replaced
        let t = vrndnq_f64(vmulq_f64(x, pd(S)));
        let jt = vcvtq_s64_f64(t);
        let (j0, j1) = (vgetq_lane_s64::<0>(jt), vgetq_lane_s64::<1>(jt));
        let (a0, a1) = (((j0 >> 6) & 0x3f) as usize, ((j1 >> 6) & 0x3f) as usize);
        let (b0, b1) = ((j0 & 0x3f) as usize, (j1 & 0x3f) as usize);
        let xl = vld1q_f64([T0[a0][0], T0[a1][0]].as_ptr());
        let xh = vld1q_f64([T0[a0][1], T0[a1][1]].as_ptr());
        let cl = vld1q_f64([T1[b0][0], T1[b1][0]].as_ptr());
        let ch = vld1q_f64([T1[b0][1], T1[b1][1]].as_ptr());
        let th = vmulq_f64(ch, xh);
        // vfmaq_f64(a, b, c) is a + b * c, rounded once: mul_add(ch, xh, -th)
        let tl = vaddq_f64(
            vaddq_f64(vmulq_f64(ch, xl), vmulq_f64(cl, xh)),
            vfmaq_f64(vnegq_f64(th), ch, xh),
        );
        // the fast path, as exp_body has it
        let dx = vaddq_f64(vsubq_f64(x, vmulq_f64(pd(L2H), t)), vmulq_f64(pd(L2L), t));
        let dx2 = vmulq_f64(dx, dx);
        let p = vaddq_f64(
            vaddq_f64(pd(FAST_C[0]), vmulq_f64(dx, pd(FAST_C[1]))),
            vmulq_f64(dx2, vaddq_f64(pd(FAST_C[2]), vmulq_f64(dx, pd(FAST_C[3])))),
        );
        let tx = vmulq_f64(th, dx);
        let fl = vaddq_f64(tl, vmulq_f64(tx, p));
        let ub = vaddq_f64(th, vaddq_f64(fl, pd(EPS)));
        let lb = vaddq_f64(th, vsubq_f64(fl, pd(EPS)));
        // ldexp_bits(lb, ie), ie = jt >> 12
        let scale = vshlq_n_u64::<52>(vreinterpretq_u64_s64(vshrq_n_s64::<12>(jt)));
        let result = vreinterpretq_f64_u64(vaddq_u64(vreinterpretq_u64_f64(lb), scale));
        vst1q_f64(dst.add(i), result);
        let rounded = vandq_u64(vceqq_f64(ub, lb), ordinary);
        let lanes = [
            (vgetq_lane_u64::<0>(rounded), vgetq_lane_f64::<0>(x)),
            (vgetq_lane_u64::<1>(rounded), vgetq_lane_f64::<1>(x)),
        ];
        for (lane, (ok, v)) in lanes.into_iter().enumerate() {
            if ok == 0 {
                *dst.add(i + lane) = exp_body(v);
                redone += 1;
            }
        }
    }
    for i in pairs..len {
        *dst.add(i) = exp_body(*src.add(i));
    }
    redone + (len - pairs)
}

/// `1 / (1 + exp(-v))` for each of `values`, in place: a negation pass, [`exp_in_place`], then the
/// division pass. Each step is correctly rounded value by value, so the bits are the expression's
/// as one value at a time computes it.
pub fn sigmoid_in_place(values: &mut [f64]) {
    for v in values.iter_mut() {
        *v = -*v;
    }
    exp_in_place(values);
    for v in values.iter_mut() {
        *v = 1.0 / (1.0 + *v);
    }
}

// x + y and its rounding error, for |x| >= |y| (or x = 0)
#[inline(always)]
fn fast_two_sum(x: f64, y: f64) -> (f64, f64) {
    let s = x + y;
    let z = s - x;
    (s, y - z)
}

#[inline(always)]
fn fast_sum(xh: f64, xl: f64, yh: f64, yl: f64) -> (f64, f64) {
    let (sh, sl) = fast_two_sum(xh, yh);
    (sh, (xl + yl) + sl)
}

// (xh + xl) * (ch + cl) as a double-double
#[inline(always)]
fn mul_dd(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let ahhh = ch * xh;
    (ahhh, (ch * xl + cl * xh) + ch.mul_add(xh, -ahhh))
}

// (ah + al) * (bh + bl) + (ch + cl): CORE-MATH's FastFMA_DW (Jeannerod, Joldes, Louvet and
// Muller's Algorithm 4, its last two FMAs as a multiply and an add, as the source has them)
#[inline(always)]
fn fast_fma_dw(ah: f64, al: f64, bh: f64, bl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let dh = ah.mul_add(bh, ch);
    let t = ch - dh;
    let e = ah.mul_add(bh, t);
    let f = e + cl;
    let g = ah * bl + f;
    (dh, al * bh + g)
}

// the polynomial c[0] + c[1] x + ... at the double-double x, by Horner in double-double
#[inline(always)]
fn poly_dd(xh: f64, xl: f64, c: &[[f64; 2]]) -> (f64, f64) {
    let (mut ch, mut cl) = (c[c.len() - 1][0], c[c.len() - 1][1]);
    for coefficient in c[..c.len() - 1].iter().rev() {
        (ch, cl) = fast_fma_dw(xh, xl, ch, cl, coefficient[0], coefficient[1]);
    }
    (ch, cl)
}

// x * 2^i by the exponent's bits (the result stays normal where it is used)
#[inline(always)]
fn ldexp_bits(x: f64, i: i64) -> f64 {
    f64::from_bits(x.to_bits().wrapping_add((i as u64) << 52))
}

// x's exponent field set to 0: the subnormal it encodes
#[inline(always)]
fn to_denormal(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & (u64::MAX >> 12))
}

const S: f64 = f64::from_bits(0x40b71547652b82fe); // 0x1.71547652b82fep+12, 2^12 / log(2)
const L2H: f64 = f64::from_bits(0x3f262e42ff000000); // 0x1.62e42ffp-13
const L2L: f64 = f64::from_bits(0x3d0718432a1b0e26); // 0x1.718432a1b0e26p-47
const L2LL: f64 = f64::from_bits(0x3999ff0342542fc3); // 0x1.9ff0342542fc3p-102
const EPS: f64 = 1.64e-19;
// x < -0x1.6232bdd7abcd2p+9: the result is subnormal
const SUBNORMAL_BELOW: u64 = 0xc086232bdd7abcd2;

// 1 + dx (c0 + c1 dx + c2 dx^2 + c3 dx^3) approximates exp(dx) to 2^-76.173 (Sollya)
const FAST_C: [f64; 4] = [
    1.0,
    0.5,
    f64::from_bits(0x3fc55555557e54ff), // 0x1.55555557e54ffp-3
    f64::from_bits(0x3fa55555553a12f4), // 0x1.55555553a12f4p-5
];

#[inline(always)]
fn reduce(x: f64) -> (f64, i64, f64, f64) {
    let t = (x * S).round_ties_even();
    let jt = t as i64;
    let (i0, i1, ie) = (((jt >> 6) & 0x3f) as usize, (jt & 0x3f) as usize, jt >> 12);
    let (th, tl) = mul_dd(T0[i0][1], T0[i0][0], T1[i1][1], T1[i1][0]);
    (t, ie, th, tl)
}

#[inline(always)]
fn exp_body(x: f64) -> f64 {
    let bits = x.to_bits();
    let abs_bits = bits & (u64::MAX >> 1);
    if abs_bits <= 0x3c90000000000000 {
        // |x| <= 2^-54: exp(x) rounds to 1 + x
        return 1.0 + x;
    }
    if abs_bits >= 0x40862e42fefa39f0 {
        // |x| >= 0x1.62e42fefa39fp+9
        if abs_bits > 0x7ff0000000000000 {
            return x + x; // NaN
        }
        if abs_bits == 0x7ff0000000000000 {
            return if bits >> 63 == 1 { 0.0 } else { x };
        }
        if bits >> 63 == 0 {
            return f64::INFINITY;
        }
        if abs_bits >= 0x40874910d52d3052 {
            return 0.0; // x <= -0x1.74910d52d3052p+9
        }
    }
    let (t, ie, th, tl) = reduce(x);
    // Cody-Waite: |t| < 2^23 and L2H has 29 bits, so L2H * t is exact
    let dx = (x - L2H * t) + L2L * t;
    let dx2 = dx * dx;
    let p = (FAST_C[0] + dx * FAST_C[1]) + dx2 * (FAST_C[2] + dx * FAST_C[3]);
    let mut fh = th;
    let tx = th * dx;
    let mut fl = tl + tx * p;
    if bits > SUBNORMAL_BELOW {
        let (h, e) = fast_two_sum(f64::from_bits(((1 - ie) as u64) << 52), fh);
        fh = h;
        fl += e;
        let (ub, lb) = (fh + (fl + EPS), fh + (fl - EPS));
        if ub != lb {
            return exp_accurate(x);
        }
        to_denormal(lb)
    } else {
        let (ub, lb) = (fh + (fl + EPS), fh + (fl - EPS));
        if ub != lb {
            return exp_accurate(x);
        }
        ldexp_bits(lb, ie)
    }
}

// the rounding test failed: double-double throughout; |x| > 2^-54 here
#[cold]
#[inline(never)]
fn exp_accurate(x: f64) -> f64 {
    // ch[0] x + ch[1] x^2 + ... + ch[6] x^7 approximates exp(x) - 1 for |x| < 2^-13.528
    const CH: [[f64; 2]; 7] = [
        [1.0, 0.0],
        [0.5, f64::from_bits(0x39c712f72ecec2cf)], // 0x1p-1, 0x1.712f72ecec2cfp-99
        [f64::from_bits(0x3fc5555555555555), f64::from_bits(0x3c65555555554d07)], // 0x1.5555555555555p-3, 0x1.5555555554d07p-57
        [f64::from_bits(0x3fa5555555555555), f64::from_bits(0x3c455194d28275da)], // 0x1.5555555555555p-5, 0x1.55194d28275dap-59
        [f64::from_bits(0x3f81111111111111), f64::from_bits(0x3c012faa0e1c0f7b)], // 0x1.1111111111111p-7, 0x1.12faa0e1c0f7bp-63
        [f64::from_bits(0x3f56c16c16da6973), f64::from_bits(0xbbf4ba45ab25d2a3)], // 0x1.6c16c16da6973p-10, -0x1.4ba45ab25d2a3p-64
        [f64::from_bits(0x3f2a01a019eb7f31), f64::from_bits(0xbbc9091d845ecd36)], // 0x1.a01a019eb7f31p-13, -0x1.9091d845ecd36p-67
    ];
    let bits = x.to_bits();
    let (t, ie, th, tl) = reduce(x);
    let dx = x - L2H * t;
    let dxl = L2L * t;
    let dxll = L2LL * t + L2L.mul_add(t, -dxl);
    let (dxh, dxl) = fast_two_sum(dx, dxl);
    let dxl = dxl + dxll;
    let (fh, fl) = poly_dd(dxh, dxl, &CH);
    let (fh, fl) = mul_dd(dxh, dxl, fh, fl);
    let (fh, fl) = mul_dd(fh, fl, th, tl);
    let (fh, fl) = fast_sum(th, tl, fh, fl);
    if bits > SUBNORMAL_BELOW {
        let (h, e) = fast_two_sum(f64::from_bits(((1 - ie) as u64) << 52), fh);
        to_denormal(h + (fl + e))
    } else {
        let (fh, fl) = fast_two_sum(fh, fl);
        // the low part within 4 ulps of a power-of-two boundary pattern: maybe a database input
        let d = fl.to_bits().wrapping_add(4) & (u64::MAX >> 12);
        let fh = if d <= 4 { from_database(x, fh) } else { fh };
        ldexp_bits(fh, ie)
    }
}

// the inputs the accurate path alone doesn't round correctly (in some rounding mode), by binary
// search; the result before scaling by 2^ie
fn from_database(x: f64, f: f64) -> f64 {
    let (mut a, mut b) = (0, EXCEPTIONS.len());
    while a + 1 < b {
        let m = (a + b) / 2;
        if x < EXCEPTIONS[m][0] {
            b = m;
        } else {
            a = m;
        }
    }
    if x == EXCEPTIONS[a][0] {
        EXCEPTIONS[a][1] + EXCEPTIONS[a][2]
    } else {
        f
    }
}

// EXCEPTIONS: x, then exp(x)'s double-double (before 2^ie), sorted by x. T0[i] and T1[i]: 2^(i/2^6)
// and 2^(i/2^12) as double-doubles, low part first (some nudged off round-to-nearest to minimise
// the exceptions, CORE-MATH's exp.sage).
const EXCEPTIONS: [[f64; 3]; 21] = [
    [
        f64::from_bits(0xc038f80e06f3a04c),
        f64::from_bits(0x3fff80aafa92b498),
        f64::from_bits(0xb95191595b7e88ed),
    ], // -0x1.8f80e06f3a04cp+4, 0x1.f80aafa92b498p+0, -0x1.191595b7e88edp-106
    [
        f64::from_bits(0xc01daf3afee51991),
        f64::from_bits(0x3ff39c6240f78d70),
        f64::from_bits(0xb962b74a3833bf10),
    ], // -0x1.daf3afee51991p+2, 0x1.39c6240f78d7p+0, -0x1.2b74a3833bf1p-105
    [
        f64::from_bits(0xbfd07e05e6e03b83),
        f64::from_bits(0x3ff8bb0d03b36f1c),
        f64::from_bits(0x3c9ffffffffffffe),
    ], // -0x1.07e05e6e03b83p-2, 0x1.8bb0d03b36f1cp+0, 0x1.ffffffffffffep-54
    [
        f64::from_bits(0xbfbdaf693d64fada),
        f64::from_bits(0x3ffc7f14af0a08eb),
        f64::from_bits(0xb932aa7e4ef70195),
    ], // -0x1.daf693d64fadap-4, 0x1.c7f14af0a08ebp+0, -0x1.2aa7e4ef70195p-108
    [
        f64::from_bits(0xbfba8f783d749a8f),
        f64::from_bits(0x3ffcd8abd4de5c33),
        f64::from_bits(0x3c9fffffffffffff),
    ], // -0x1.a8f783d749a8fp-4, 0x1.cd8abd4de5c33p+0, 0x1.fffffffffffffp-54
    [
        f64::from_bits(0xbf940a990279c2c0),
        f64::from_bits(0x3fff613a44b93cdc),
        f64::from_bits(0x3c9ffffffffffffd),
    ], // -0x1.40a990279c2cp-6, 0x1.f613a44b93cdcp+0, 0x1.ffffffffffffdp-54
    [
        f64::from_bits(0xbf2a2fefefd580df),
        f64::from_bits(0x3ffffe5d0bb7eabf),
        f64::from_bits(0x39081905a57d6a44),
    ], // -0x1.a2fefefd580dfp-13, 0x1.ffe5d0bb7eabfp+0, 0x1.81905a57d6a44p-111
    [
        f64::from_bits(0xbf1664716b68a409),
        f64::from_bits(0x3fffff4cde6a0bfb),
        f64::from_bits(0xbc9fffffffffffff),
    ], // -0x1.664716b68a409p-14, 0x1.fff4cde6a0bfbp+0, -0x1.fffffffffffffp-54
    [
        f64::from_bits(0xbe30401ae48409b5),
        f64::from_bits(0x3feffffffdf7fca3),
        f64::from_bits(0x3c8fffffffffffff),
    ], // -0x1.0401ae48409b5p-28, 0x1.ffffffdf7fca3p-1, 0x1.fffffffffffffp-55
    [
        f64::from_bits(0xbe1ff171507f8ba5),
        f64::from_bits(0x3fefffffff007475),
        f64::from_bits(0x3c8fffffffffffff),
    ], // -0x1.ff171507f8ba5p-30, 0x1.fffffff007475p-1, 0x1.fffffffffffffp-55
    [
        f64::from_bits(0xbe185e60704a3a9c),
        f64::from_bits(0x3fefffffff3d0cfd),
        f64::from_bits(0xbc8fffffffffffff),
    ], // -0x1.85e60704a3a9cp-30, 0x1.fffffff3d0cfdp-1, -0x1.fffffffffffffp-55
    [
        f64::from_bits(0xbe04bd46601ae1ef),
        f64::from_bits(0x3fefffffffad0ae7),
        f64::from_bits(0xbc8fffffffffffff),
    ], // -0x1.4bd46601ae1efp-31, 0x1.fffffffad0ae7p-1, -0x1.fffffffffffffp-55
    [
        f64::from_bits(0x3ca0000000000000),
        f64::from_bits(0x3ff0000000000001),
        f64::from_bits(0xbc9fffffffffffff),
    ], // 0x1p-53, 0x1.0000000000001p+0, -0x1.fffffffffffffp-54
    [
        f64::from_bits(0x3dbffff7fffe0001),
        f64::from_bits(0x3ff000000001ffff),
        f64::from_bits(0x3c9fffffffffffff),
    ], // 0x1.ffff7fffe0001p-36, 0x1.000000001ffffp+0, 0x1.fffffffffffffp-54
    [
        f64::from_bits(0x3dd8cd99fffb319c),
        f64::from_bits(0x3ff0000000063367),
        f64::from_bits(0xbc9fffffffffffff),
    ], // 0x1.8cd99fffb319cp-34, 0x1.0000000063367p+0, -0x1.fffffffffffffp-54
    [
        f64::from_bits(0x3e0413f33fe6ce07),
        f64::from_bits(0x3ff00000002827e7),
        f64::from_bits(0xbc9fffffffffffff),
    ], // 0x1.413f33fe6ce07p-31, 0x1.00000002827e7p+0, -0x1.fffffffffffffp-54
    [
        f64::from_bits(0x3e1f8e165f8388f7),
        f64::from_bits(0x3ff00000007e3859),
        f64::from_bits(0x3c9fffffffffffff),
    ], // 0x1.f8e165f8388f7p-30, 0x1.00000007e3859p+0, 0x1.fffffffffffffp-54
    [
        f64::from_bits(0x3eb282aae3169ee7),
        f64::from_bits(0x3ff00001282ab8e7),
        f64::from_bits(0xbc9fffffffffffff),
    ], // 0x1.282aae3169ee7p-20, 0x1.00001282ab8e7p+0, -0x1.fffffffffffffp-54
    [
        f64::from_bits(0x3ebe5ba92aa9b1ef),
        f64::from_bits(0x3ff00001e5baaf77),
        f64::from_bits(0x3c9ffffffffffffe),
    ], // 0x1.e5ba92aa9b1efp-20, 0x1.00001e5baaf77p+0, 0x1.ffffffffffffep-54
    [
        f64::from_bits(0x3f76a4d1af9cc989),
        f64::from_bits(0x3ff016b4df3299d7),
        f64::from_bits(0x3c9ffffffffffffe),
    ], // 0x1.6a4d1af9cc989p-8, 0x1.016b4df3299d7p+0, 0x1.ffffffffffffep-54
    [
        f64::from_bits(0x3ffaca7ae8da5a7b),
        f64::from_bits(0x3ff557d4acd7e557),
        f64::from_bits(0xbc9fffffffffffff),
    ], // 0x1.aca7ae8da5a7bp+0, 0x1.557d4acd7e557p+0, -0x1.fffffffffffffp-54
];

const T0: [[f64; 2]; 64] = [
    [f64::from_bits(0x0000000000000000), f64::from_bits(0x3ff0000000000000)], // 0x0p+0, 0x1p+0
    [f64::from_bits(0xbc719083535b085c), f64::from_bits(0x3ff02c9a3e778061)], // -0x1.19083535b085cp-56, 0x1.02c9a3e778061p+0
    [f64::from_bits(0x3c8d73e2a475b466), f64::from_bits(0x3ff059b0d3158574)], // 0x1.d73e2a475b466p-55, 0x1.059b0d3158574p+0
    [f64::from_bits(0x3c6186be4bb284fd), f64::from_bits(0x3ff0874518759bc8)], // 0x1.186be4bb284fdp-57, 0x1.0874518759bc8p+0
    [f64::from_bits(0x3c98a62e4adc610b), f64::from_bits(0x3ff0b5586cf9890f)], // 0x1.8a62e4adc610bp-54, 0x1.0b5586cf9890fp+0
    [f64::from_bits(0x3c403a1727c57b55), f64::from_bits(0x3ff0e3ec32d3d1a2)], // 0x1.03a1727c57b55p-59, 0x1.0e3ec32d3d1a2p+0
    [f64::from_bits(0xbc96c51039449b3a), f64::from_bits(0x3ff11301d0125b51)], // -0x1.6c51039449b3ap-54, 0x1.11301d0125b51p+0
    [f64::from_bits(0xbc932fbf9af1369e), f64::from_bits(0x3ff1429aaea92de0)], // -0x1.32fbf9af1369ep-54, 0x1.1429aaea92dep+0
    [f64::from_bits(0xbc819041b9d78a76), f64::from_bits(0x3ff172b83c7d517b)], // -0x1.19041b9d78a76p-55, 0x1.172b83c7d517bp+0
    [f64::from_bits(0x3c8e5b4c7b4968e4), f64::from_bits(0x3ff1a35beb6fcb75)], // 0x1.e5b4c7b4968e4p-55, 0x1.1a35beb6fcb75p+0
    [f64::from_bits(0x3c9e016e00a2643c), f64::from_bits(0x3ff1d4873168b9aa)], // 0x1.e016e00a2643cp-54, 0x1.1d4873168b9aap+0
    [f64::from_bits(0x3c8dc775814a8496), f64::from_bits(0x3ff2063b88628cd6)], // 0x1.dc775814a8496p-55, 0x1.2063b88628cd6p+0
    [f64::from_bits(0x3c99b07eb6c70572), f64::from_bits(0x3ff2387a6e756238)], // 0x1.9b07eb6c70572p-54, 0x1.2387a6e756238p+0
    [f64::from_bits(0x3c82bd339940e9da), f64::from_bits(0x3ff26b4565e27cdd)], // 0x1.2bd339940e9dap-55, 0x1.26b4565e27cddp+0
    [f64::from_bits(0x3c8612e8afad1255), f64::from_bits(0x3ff29e9df51fdee1)], // 0x1.612e8afad1255p-55, 0x1.29e9df51fdee1p+0
    [f64::from_bits(0x3c90024754db41d4), f64::from_bits(0x3ff2d285a6e4030b)], // 0x1.0024754db41d4p-54, 0x1.2d285a6e4030bp+0
    [f64::from_bits(0x3c86f46ad23182e4), f64::from_bits(0x3ff306fe0a31b715)], // 0x1.6f46ad23182e4p-55, 0x1.306fe0a31b715p+0
    [f64::from_bits(0x3c932721843659a6), f64::from_bits(0x3ff33c08b26416ff)], // 0x1.32721843659a6p-54, 0x1.33c08b26416ffp+0
    [f64::from_bits(0xbc963aeabf42eae0), f64::from_bits(0x3ff371a7373aa9cb)], // -0x1.63aeabf42eaep-54, 0x1.371a7373aa9cbp+0
    [f64::from_bits(0xbc75e436d661f5db), f64::from_bits(0x3ff3a7db34e59ff7)], // -0x1.5e436d661f5dbp-56, 0x1.3a7db34e59ff7p+0
    [f64::from_bits(0x3c8ada0911f09eba), f64::from_bits(0x3ff3dea64c123422)], // 0x1.ada0911f09ebap-55, 0x1.3dea64c123422p+0
    [f64::from_bits(0xbc5ef3691c309278), f64::from_bits(0x3ff4160a21f72e2a)], // -0x1.ef3691c309278p-58, 0x1.4160a21f72e2ap+0
    [f64::from_bits(0x3c489b7a04ef80d0), f64::from_bits(0x3ff44e086061892d)], // 0x1.89b7a04ef80dp-59, 0x1.44e086061892dp+0
    [f64::from_bits(0x3c73c1a3b69062f0), f64::from_bits(0x3ff486a2b5c13cd0)], // 0x1.3c1a3b69062fp-56, 0x1.486a2b5c13cdp+0
    [f64::from_bits(0x3c7d4397afec42e2), f64::from_bits(0x3ff4bfdad5362a27)], // 0x1.d4397afec42e2p-56, 0x1.4bfdad5362a27p+0
    [f64::from_bits(0xbc94b309d25957e3), f64::from_bits(0x3ff4f9b2769d2ca7)], // -0x1.4b309d25957e3p-54, 0x1.4f9b2769d2ca7p+0
    [f64::from_bits(0xbc807abe1db13cab), f64::from_bits(0x3ff5342b569d4f82)], // -0x1.07abe1db13cabp-55, 0x1.5342b569d4f82p+0
    [f64::from_bits(0x3c99bb2c011d93ac), f64::from_bits(0x3ff56f4736b527da)], // 0x1.9bb2c011d93acp-54, 0x1.56f4736b527dap+0
    [f64::from_bits(0x3c96324c054647ac), f64::from_bits(0x3ff5ab07dd485429)], // 0x1.6324c054647acp-54, 0x1.5ab07dd485429p+0
    [f64::from_bits(0x3c9ba6f93080e65e), f64::from_bits(0x3ff5e76f15ad2148)], // 0x1.ba6f93080e65ep-54, 0x1.5e76f15ad2148p+0
    [f64::from_bits(0xbc9383c17e40b497), f64::from_bits(0x3ff6247eb03a5585)], // -0x1.383c17e40b497p-54, 0x1.6247eb03a5585p+0
    [f64::from_bits(0xbc9bb60987591c34), f64::from_bits(0x3ff6623882552225)], // -0x1.bb60987591c34p-54, 0x1.6623882552225p+0
    [f64::from_bits(0xbc9bdd3413b26456), f64::from_bits(0x3ff6a09e667f3bcd)], // -0x1.bdd3413b26456p-54, 0x1.6a09e667f3bcdp+0
    [f64::from_bits(0xbc6bbe3a683c88aa), f64::from_bits(0x3ff6dfb23c651a2f)], // -0x1.bbe3a683c88aap-57, 0x1.6dfb23c651a2fp+0
    [f64::from_bits(0xbc816e4786887a9a), f64::from_bits(0x3ff71f75e8ec5f74)], // -0x1.16e4786887a9ap-55, 0x1.71f75e8ec5f74p+0
    [f64::from_bits(0xbc90245957316dd4), f64::from_bits(0x3ff75feb564267c9)], // -0x1.0245957316dd4p-54, 0x1.75feb564267c9p+0
    [f64::from_bits(0xbc841577ee049930), f64::from_bits(0x3ff7a11473eb0187)], // -0x1.41577ee04993p-55, 0x1.7a11473eb0187p+0
    [f64::from_bits(0x3c705d02ba15797e), f64::from_bits(0x3ff7e2f336cf4e62)], // 0x1.05d02ba15797ep-56, 0x1.7e2f336cf4e62p+0
    [f64::from_bits(0xbc9d4c1dd41532d8), f64::from_bits(0x3ff82589994cce13)], // -0x1.d4c1dd41532d8p-54, 0x1.82589994cce13p+0
    [f64::from_bits(0xbc9fc6f89bd4f6ba), f64::from_bits(0x3ff868d99b4492ed)], // -0x1.fc6f89bd4f6bap-54, 0x1.868d99b4492edp+0
    [f64::from_bits(0x3c96e9f156864b29), f64::from_bits(0x3ff8ace5422aa0db)], // 0x1.6e9f156864b29p-54, 0x1.8ace5422aa0dbp+0
    [f64::from_bits(0x3c85cc13a2e3976f), f64::from_bits(0x3ff8f1ae99157736)], // 0x1.5cc13a2e3976fp-55, 0x1.8f1ae99157736p+0
    [f64::from_bits(0xbc675fc781b57ebc), f64::from_bits(0x3ff93737b0cdc5e5)], // -0x1.75fc781b57ebcp-57, 0x1.93737b0cdc5e5p+0
    [f64::from_bits(0xbc9d185b7c1b85d0), f64::from_bits(0x3ff97d829fde4e50)], // -0x1.d185b7c1b85dp-54, 0x1.97d829fde4e5p+0
    [f64::from_bits(0x3c7c7c46b071f2be), f64::from_bits(0x3ff9c49182a3f090)], // 0x1.c7c46b071f2bep-56, 0x1.9c49182a3f09p+0
    [f64::from_bits(0xbc9359495d1cd532), f64::from_bits(0x3ffa0c667b5de565)], // -0x1.359495d1cd532p-54, 0x1.a0c667b5de565p+0
    [f64::from_bits(0xbc9d2f6edb8d41e3), f64::from_bits(0x3ffa5503b23e255d)], // -0x1.d2f6edb8d41e3p-54, 0x1.a5503b23e255dp+0
    [f64::from_bits(0x3c90fac90ef7fd32), f64::from_bits(0x3ffa9e6b5579fdbf)], // 0x1.0fac90ef7fd32p-54, 0x1.a9e6b5579fdbfp+0
    [f64::from_bits(0x3c97a1cd345dcc82), f64::from_bits(0x3ffae89f995ad3ad)], // 0x1.7a1cd345dcc82p-54, 0x1.ae89f995ad3adp+0
    [f64::from_bits(0xbc62805e3084d708), f64::from_bits(0x3ffb33a2b84f15fb)], // -0x1.2805e3084d708p-57, 0x1.b33a2b84f15fbp+0
    [f64::from_bits(0xbc75584f7e54ac3b), f64::from_bits(0x3ffb7f76f2fb5e47)], // -0x1.5584f7e54ac3bp-56, 0x1.b7f76f2fb5e47p+0
    [f64::from_bits(0x3c823dd07a2d9e84), f64::from_bits(0x3ffbcc1e904bc1d2)], // 0x1.23dd07a2d9e84p-55, 0x1.bcc1e904bc1d2p+0
    [f64::from_bits(0x3c811065895048de), f64::from_bits(0x3ffc199bdd85529c)], // 0x1.11065895048dep-55, 0x1.c199bdd85529cp+0
    [f64::from_bits(0x3c92884dff483cad), f64::from_bits(0x3ffc67f12e57d14b)], // 0x1.2884dff483cadp-54, 0x1.c67f12e57d14bp+0
    [f64::from_bits(0x3c7503cbd1e949d9), f64::from_bits(0x3ffcb720dcef9069)], // 0x1.503cbd1e949d9p-56, 0x1.cb720dcef9069p+0
    [f64::from_bits(0xbc9cbc3743797a9c), f64::from_bits(0x3ffd072d4a07897c)], // -0x1.cbc3743797a9cp-54, 0x1.d072d4a07897cp+0
    [f64::from_bits(0x3c82ed02d75b3706), f64::from_bits(0x3ffd5818dcfba487)], // 0x1.2ed02d75b3706p-55, 0x1.d5818dcfba487p+0
    [f64::from_bits(0x3c9c2300696db533), f64::from_bits(0x3ffda9e603db3285)], // 0x1.c2300696db533p-54, 0x1.da9e603db3285p+0
    [f64::from_bits(0xbc91a5cd4f184b5c), f64::from_bits(0x3ffdfc97337b9b5f)], // -0x1.1a5cd4f184b5cp-54, 0x1.dfc97337b9b5fp+0
    [f64::from_bits(0x3c839e8980a9cc90), f64::from_bits(0x3ffe502ee78b3ff6)], // 0x1.39e8980a9cc9p-55, 0x1.e502ee78b3ff6p+0
    [f64::from_bits(0xbc9e9c23179c2894), f64::from_bits(0x3ffea4afa2a490da)], // -0x1.e9c23179c2894p-54, 0x1.ea4afa2a490dap+0
    [f64::from_bits(0x3c9dc7f486a4b6b1), f64::from_bits(0x3ffefa1bee615a27)], // 0x1.dc7f486a4b6b1p-54, 0x1.efa1bee615a27p+0
    [f64::from_bits(0x3c99d3e12dd8a18d), f64::from_bits(0x3fff50765b6e4540)], // 0x1.9d3e12dd8a18dp-54, 0x1.f50765b6e454p+0
    [f64::from_bits(0x3c874853f3a5931f), f64::from_bits(0x3fffa7c1819e90d8)], // 0x1.74853f3a5931fp-55, 0x1.fa7c1819e90d8p+0
];

const T1: [[f64; 2]; 64] = [
    [f64::from_bits(0x0000000000000000), f64::from_bits(0x3ff0000000000000)], // 0x0p+0, 0x1p+0
    [f64::from_bits(0x3c9ae8e38c59c72c), f64::from_bits(0x3ff000b175effdc7)], // 0x1.ae8e38c59c72cp-54, 0x1.000b175effdc7p+0
    [f64::from_bits(0xbc57b5d0d58ea8f4), f64::from_bits(0x3ff00162f3904052)], // -0x1.7b5d0d58ea8f4p-58, 0x1.00162f3904052p+0
    [f64::from_bits(0x3c94115cb6b16a8d), f64::from_bits(0x3ff0021478e11ce6)], // 0x1.4115cb6b16a8dp-54, 0x1.0021478e11ce6p+0
    [f64::from_bits(0xbc8d7c96f201bb2e), f64::from_bits(0x3ff002c605e2e8cf)], // -0x1.d7c96f201bb2ep-55, 0x1.002c605e2e8cfp+0
    [f64::from_bits(0x3c984711d4c35ea0), f64::from_bits(0x3ff003779a95f959)], // 0x1.84711d4c35eap-54, 0x1.003779a95f959p+0
    [f64::from_bits(0xbc80484245243778), f64::from_bits(0x3ff0042936faa3d8)], // -0x1.0484245243778p-55, 0x1.0042936faa3d8p+0
    [f64::from_bits(0xbc94b237da2025fa), f64::from_bits(0x3ff004dadb113da0)], // -0x1.4b237da2025fap-54, 0x1.004dadb113dap+0
    [f64::from_bits(0xbc75e00e62d6b30e), f64::from_bits(0x3ff0058c86da1c0a)], // -0x1.5e00e62d6b30ep-56, 0x1.0058c86da1c0ap+0
    [f64::from_bits(0x3c9a1d6cedbb9480), f64::from_bits(0x3ff0063e3a559473)], // 0x1.a1d6cedbb948p-54, 0x1.0063e3a559473p+0
    [f64::from_bits(0xbc94acf197a00141), f64::from_bits(0x3ff006eff583fc3d)], // -0x1.4acf197a00141p-54, 0x1.006eff583fc3dp+0
    [f64::from_bits(0xbc6eaf2ea42391a6), f64::from_bits(0x3ff007a1b865a8ca)], // -0x1.eaf2ea42391a6p-57, 0x1.007a1b865a8cap+0
    [f64::from_bits(0x3c7da93f90835f7a), f64::from_bits(0x3ff0085382faef83)], // 0x1.da93f90835f7ap-56, 0x1.0085382faef83p+0
    [f64::from_bits(0xbc86a79084ab0937), f64::from_bits(0x3ff00905554425d4)], // -0x1.6a79084ab0937p-55, 0x1.00905554425d4p+0
    [f64::from_bits(0x3c986364f8fbe8f8), f64::from_bits(0x3ff009b72f41a12b)], // 0x1.86364f8fbe8f8p-54, 0x1.009b72f41a12bp+0
    [f64::from_bits(0xbc882e8e14e3110b), f64::from_bits(0x3ff00a6910f3b6fd)], // -0x1.82e8e14e3110bp-55, 0x1.00a6910f3b6fdp+0
    [f64::from_bits(0xbc84f6b2a7609f72), f64::from_bits(0x3ff00b1afa5abcbf)], // -0x1.4f6b2a7609f72p-55, 0x1.00b1afa5abcbfp+0
    [f64::from_bits(0xbc7e1a258ea8f71a), f64::from_bits(0x3ff00bcceb7707ec)], // -0x1.e1a258ea8f71ap-56, 0x1.00bcceb7707ecp+0
    [f64::from_bits(0x3c74362ca5bc26f2), f64::from_bits(0x3ff00c7ee448ee02)], // 0x1.4362ca5bc26f2p-56, 0x1.00c7ee448ee02p+0
    [f64::from_bits(0x3c9095a56c919d02), f64::from_bits(0x3ff00d30e4d0c483)], // 0x1.095a56c919d02p-54, 0x1.00d30e4d0c483p+0
    [f64::from_bits(0xbc6406ac4e81a646), f64::from_bits(0x3ff00de2ed0ee0f5)], // -0x1.406ac4e81a646p-57, 0x1.00de2ed0ee0f5p+0
    [f64::from_bits(0x3c9b5a6902767e08), f64::from_bits(0x3ff00e94fd0398e0)], // 0x1.b5a6902767e08p-54, 0x1.00e94fd0398ep+0
    [f64::from_bits(0xbc991b2060859320), f64::from_bits(0x3ff00f4714af41d3)], // -0x1.91b206085932p-54, 0x1.00f4714af41d3p+0
    [f64::from_bits(0x3c8427068ab22306), f64::from_bits(0x3ff00ff93412315c)], // 0x1.427068ab22306p-55, 0x1.00ff93412315cp+0
    [f64::from_bits(0x3c9c1d0660524e08), f64::from_bits(0x3ff010ab5b2cbd11)], // 0x1.c1d0660524e08p-54, 0x1.010ab5b2cbd11p+0
    [f64::from_bits(0xbc9e7bdfb3204be8), f64::from_bits(0x3ff0115d89ff3a8b)], // -0x1.e7bdfb3204be8p-54, 0x1.0115d89ff3a8bp+0
    [f64::from_bits(0x3c8843aa8b9cbbc6), f64::from_bits(0x3ff0120fc089ff63)], // 0x1.843aa8b9cbbc6p-55, 0x1.0120fc089ff63p+0
    [f64::from_bits(0xbc734104ee7edae8), f64::from_bits(0x3ff012c1fecd613b)], // -0x1.34104ee7edae8p-56, 0x1.012c1fecd613bp+0
    [f64::from_bits(0xbc72b6aeb6176892), f64::from_bits(0x3ff0137444c9b5b5)], // -0x1.2b6aeb6176892p-56, 0x1.0137444c9b5b5p+0
    [f64::from_bits(0x3c7a8cd33b8a1bb3), f64::from_bits(0x3ff01426927f5278)], // 0x1.a8cd33b8a1bb3p-56, 0x1.01426927f5278p+0
    [f64::from_bits(0x3c72edc08e5da99a), f64::from_bits(0x3ff014d8e7ee8d2f)], // 0x1.2edc08e5da99ap-56, 0x1.014d8e7ee8d2fp+0
    [f64::from_bits(0x3c857ba2dc7e0c72), f64::from_bits(0x3ff0158b4517bb88)], // 0x1.57ba2dc7e0c72p-55, 0x1.0158b4517bb88p+0
    [f64::from_bits(0x3c9b61299ab8cdb8), f64::from_bits(0x3ff0163da9fb3335)], // 0x1.b61299ab8cdb8p-54, 0x1.0163da9fb3335p+0
    [f64::from_bits(0xbc990565902c5f43), f64::from_bits(0x3ff016f0169949ed)], // -0x1.90565902c5f43p-54, 0x1.016f0169949edp+0
    [f64::from_bits(0x3c870fc41c5c2d54), f64::from_bits(0x3ff017a28af25567)], // 0x1.70fc41c5c2d54p-55, 0x1.017a28af25567p+0
    [f64::from_bits(0x3c94b9a6e145d76c), f64::from_bits(0x3ff018550706ab62)], // 0x1.4b9a6e145d76cp-54, 0x1.018550706ab62p+0
    [f64::from_bits(0xbc7008eff5142bfa), f64::from_bits(0x3ff019078ad6a19f)], // -0x1.008eff5142bfap-56, 0x1.019078ad6a19fp+0
    [f64::from_bits(0xbc977669f033c7de), f64::from_bits(0x3ff019ba16628de2)], // -0x1.77669f033c7dep-54, 0x1.019ba16628de2p+0
    [f64::from_bits(0xbc909bb78eeead0a), f64::from_bits(0x3ff01a6ca9aac5f3)], // -0x1.09bb78eeead0ap-54, 0x1.01a6ca9aac5f3p+0
    [f64::from_bits(0x3c9371231477ece4), f64::from_bits(0x3ff01b1f44af9f9e)], // 0x1.371231477ece4p-54, 0x1.01b1f44af9f9ep+0
    [f64::from_bits(0x3c75e7626621eb5a), f64::from_bits(0x3ff01bd1e77170b4)], // 0x1.5e7626621eb5ap-56, 0x1.01bd1e77170b4p+0
    [f64::from_bits(0xbc9bc72b100828a4), f64::from_bits(0x3ff01c8491f08f08)], // -0x1.bc72b100828a4p-54, 0x1.01c8491f08f08p+0
    [f64::from_bits(0xbc6ce39cbbab8bbe), f64::from_bits(0x3ff01d37442d5070)], // -0x1.ce39cbbab8bbep-57, 0x1.01d37442d507p+0
    [f64::from_bits(0x3c816996709da2e5), f64::from_bits(0x3ff01de9fe280ac8)], // 0x1.16996709da2e5p-55, 0x1.01de9fe280ac8p+0
    [f64::from_bits(0xbc8c11f5239bf536), f64::from_bits(0x3ff01e9cbfe113ef)], // -0x1.c11f5239bf536p-55, 0x1.01e9cbfe113efp+0
    [f64::from_bits(0x3c8e1d4eb5edc6b4), f64::from_bits(0x3ff01f4f8958c1c6)], // 0x1.e1d4eb5edc6b4p-55, 0x1.01f4f8958c1c6p+0
    [f64::from_bits(0xbc9afb99946ee3f0), f64::from_bits(0x3ff020025a8f6a35)], // -0x1.afb99946ee3fp-54, 0x1.020025a8f6a35p+0
    [f64::from_bits(0xbc98f06d8a148a32), f64::from_bits(0x3ff020b533856324)], // -0x1.8f06d8a148a32p-54, 0x1.020b533856324p+0
    [f64::from_bits(0xbc82bf310fc54eb8), f64::from_bits(0x3ff02168143b0281)], // -0x1.2bf310fc54eb8p-55, 0x1.02168143b0281p+0
    [f64::from_bits(0xbc9c95a035eb4177), f64::from_bits(0x3ff0221afcb09e3e)], // -0x1.c95a035eb4177p-54, 0x1.0221afcb09e3ep+0
    [f64::from_bits(0xbc9491793e46834c), f64::from_bits(0x3ff022cdece68c4f)], // -0x1.491793e46834cp-54, 0x1.022cdece68c4fp+0
    [f64::from_bits(0xbc73e8d0d9c4908f), f64::from_bits(0x3ff02380e4dd22ad)], // -0x1.3e8d0d9c4908fp-56, 0x1.02380e4dd22adp+0
    [f64::from_bits(0xbc9314aa16278aa4), f64::from_bits(0x3ff02433e494b755)], // -0x1.314aa16278aa4p-54, 0x1.02433e494b755p+0
    [f64::from_bits(0x3c848daf888e9650), f64::from_bits(0x3ff024e6ec0da046)], // 0x1.48daf888e965p-55, 0x1.024e6ec0da046p+0
    [f64::from_bits(0x3c856dc8046821f4), f64::from_bits(0x3ff02599fb483385)], // 0x1.56dc8046821f4p-55, 0x1.02599fb483385p+0
    [f64::from_bits(0x3c945b42356b9d46), f64::from_bits(0x3ff0264d1244c719)], // 0x1.45b42356b9d46p-54, 0x1.0264d1244c719p+0
    [f64::from_bits(0xbc7082ef51b61d7e), f64::from_bits(0x3ff027003103b10e)], // -0x1.082ef51b61d7ep-56, 0x1.027003103b10ep+0
    [f64::from_bits(0x3c72106ed0920a32), f64::from_bits(0x3ff027b357854772)], // 0x1.2106ed0920a32p-56, 0x1.027b357854772p+0
    [f64::from_bits(0xbc9fd4cf26ea5d11), f64::from_bits(0x3ff0286685c9e059)], // -0x1.fd4cf26ea5d11p-54, 0x1.0286685c9e059p+0
    [f64::from_bits(0xbc909f8775e78085), f64::from_bits(0x3ff02919bbd1d1d8)], // -0x1.09f8775e78085p-54, 0x1.02919bbd1d1d8p+0
    [f64::from_bits(0x3c564cbba902ca25), f64::from_bits(0x3ff029ccf99d720a)], // 0x1.64cbba902ca25p-58, 0x1.029ccf99d720ap+0
    [f64::from_bits(0x3c94383ef231d208), f64::from_bits(0x3ff02a803f2d170d)], // 0x1.4383ef231d208p-54, 0x1.02a803f2d170dp+0
    [f64::from_bits(0x3c94a47a505b3a46), f64::from_bits(0x3ff02b338c811703)], // 0x1.4a47a505b3a46p-54, 0x1.02b338c811703p+0
    [f64::from_bits(0x3c9e47120223467f), f64::from_bits(0x3ff02be6e199c811)], // 0x1.e47120223467fp-54, 0x1.02be6e199c811p+0
];
