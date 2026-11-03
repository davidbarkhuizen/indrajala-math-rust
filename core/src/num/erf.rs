//! The error function, `erf(x) = 2 / sqrt(pi) * integral from 0 to x of exp(-t^2) dt`
//! (indrajala-ml's FFN activations workplan, D2): GELU's `x * (1 + erf(x / sqrt 2)) / 2`.
//!
//! A port of fdlibm's `s_erf.c` (Sun, 1993; netlib's fdlibm 5.3, `@(#)s_erf.c 1.3 95/01/18`), its
//! `erf` half, step for step: four ranges of `|x|`, each a rational approximation, the last two
//! through `exp` twice. Its operations are `+ - * /`, `abs`, one exact truncation (the low 32 bits
//! of `|x|` cleared) and the crate's correctly rounded [`crate::num::exp::exp`] where fdlibm calls
//! `__ieee754_exp`; Rust never contracts `a * b + c`, so the bits are the expression's, the same on
//! every machine. indrajala-ml's pure-Python and numpy ports take the same steps through their own
//! `exp`, so the three agree by bits wherever their `exp` does (the tests give them the crate's).
//! Not correctly rounded: fdlibm's error is under one ulp. Constants are the C source's, from the
//! bits its comments give beside each.
//!
//! The original's notice:
//!
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunSoft, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.

use crate::num::exp::exp;

const TINY: f64 = 1e-300;
/// `c = (float)0.84506291151`, erf(1) rounded to single precision.
const ERX: f64 = f64::from_bits(0x3FEB_0AC1_6000_0000);

// erf on [0, 0.84375]: x + x * P(x^2) / Q(x^2)
const EFX: f64 = f64::from_bits(0x3FC0_6EBA_8214_DB69);
const EFX8: f64 = f64::from_bits(0x3FF0_6EBA_8214_DB69);
const PP0: f64 = f64::from_bits(0x3FC0_6EBA_8214_DB68);
const PP1: f64 = f64::from_bits(0xBFD4_CD7D_691C_B913);
const PP2: f64 = f64::from_bits(0xBF9D_2A51_DBD7_194F);
const PP3: f64 = f64::from_bits(0xBF77_A291_2366_68E4);
const PP4: f64 = f64::from_bits(0xBEF8_EAD6_1200_16AC);
const QQ1: f64 = f64::from_bits(0x3FD9_7779_CDDA_DC09);
const QQ2: f64 = f64::from_bits(0x3FB0_A54C_5536_CEBA);
const QQ3: f64 = f64::from_bits(0x3F74_D022_C4D3_6B0F);
const QQ4: f64 = f64::from_bits(0x3F21_5DC9_221C_1A10);
const QQ5: f64 = f64::from_bits(0xBED0_9C43_42A2_6120);

// erf on [0.84375, 1.25]: c + P1(s) / Q1(s), s = |x| - 1
const PA0: f64 = f64::from_bits(0xBF63_59B8_BEF7_7538);
const PA1: f64 = f64::from_bits(0x3FDA_8D00_AD92_B34D);
const PA2: f64 = f64::from_bits(0xBFD7_D240_FBB8_C3F1);
const PA3: f64 = f64::from_bits(0x3FD4_5FCA_8051_20E4);
const PA4: f64 = f64::from_bits(0xBFBC_6398_3D3E_28EC);
const PA5: f64 = f64::from_bits(0x3FA2_2A36_5997_95EB);
const PA6: f64 = f64::from_bits(0xBF61_BF38_0A96_073F);
const QA1: f64 = f64::from_bits(0x3FBB_3E66_18EE_E323);
const QA2: f64 = f64::from_bits(0x3FE1_4AF0_92EB_6F33);
const QA3: f64 = f64::from_bits(0x3FB2_635C_D99F_E9A7);
const QA4: f64 = f64::from_bits(0x3FC0_2660_E763_351F);
const QA5: f64 = f64::from_bits(0x3F8B_EDC2_6B51_DD1C);
const QA6: f64 = f64::from_bits(0x3F88_8B54_5735_151D);

// erfc on [1.25, 1/0.35]: R1(s) / S1(s), s = 1 / x^2
const RA0: f64 = f64::from_bits(0xBF84_3412_600D_6435);
const RA1: f64 = f64::from_bits(0xBFE6_3416_E4BA_7360);
const RA2: f64 = f64::from_bits(0xC025_1E04_41B0_E726);
const RA3: f64 = f64::from_bits(0xC04F_300A_E4CB_A38D);
const RA4: f64 = f64::from_bits(0xC064_4CB1_8428_2266);
const RA5: f64 = f64::from_bits(0xC067_135C_EBCC_ABB2);
const RA6: f64 = f64::from_bits(0xC054_5265_57E4_D2F2);
const RA7: f64 = f64::from_bits(0xC023_A0EF_C69A_C25C);
const SA1: f64 = f64::from_bits(0x4033_A6B9_BD70_7687);
const SA2: f64 = f64::from_bits(0x4061_350C_526A_E721);
const SA3: f64 = f64::from_bits(0x407B_290D_D58A_1A71);
const SA4: f64 = f64::from_bits(0x4084_2B19_21EC_2868);
const SA5: f64 = f64::from_bits(0x407A_D021_5770_0314);
const SA6: f64 = f64::from_bits(0x405B_28A3_EE48_AE2C);
const SA7: f64 = f64::from_bits(0x401A_47EF_8E48_4A93);
const SA8: f64 = f64::from_bits(0xBFAE_EFF2_EE74_9A62);

// erfc on [1/0.35, 6]: R2(s) / S2(s), s = 1 / x^2
const RB0: f64 = f64::from_bits(0xBF84_3412_39E8_6F4A);
const RB1: f64 = f64::from_bits(0xBFE9_93BA_70C2_85DE);
const RB2: f64 = f64::from_bits(0xC031_C209_555F_995A);
const RB3: f64 = f64::from_bits(0xC064_145D_43C5_ED98);
const RB4: f64 = f64::from_bits(0xC083_EC88_1375_F228);
const RB5: f64 = f64::from_bits(0xC090_0461_6A2E_5992);
const RB6: f64 = f64::from_bits(0xC07E_384E_9BDC_383F);
const SB1: f64 = f64::from_bits(0x403E_568B_261D_5190);
const SB2: f64 = f64::from_bits(0x4074_5CAE_221B_9F0A);
const SB3: f64 = f64::from_bits(0x4098_02EB_189D_5118);
const SB4: f64 = f64::from_bits(0x40A8_FFB7_688C_246A);
const SB5: f64 = f64::from_bits(0x40A3_F219_CEDF_3BE6);
const SB6: f64 = f64::from_bits(0x407D_A874_E79F_E763);
const SB7: f64 = f64::from_bits(0xC036_70E2_4271_2D62);

/// `erf(x)`, fdlibm's: odd, `erf(+-inf) = +-1`, `erf(NaN)` NaN, within one ulp of the exact value.
pub fn erf(x: f64) -> f64 {
    // the high word, signed as fdlibm's `int hx`, and its magnitude
    let hx = (x.to_bits() >> 32) as u32 as i32;
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        // erf(nan) = nan, erf(+-inf) = +-1
        let i = ((hx as u32) >> 31) << 1;
        return f64::from(1 - i as i32) + 1.0 / x;
    }
    if ix < 0x3feb_0000 {
        // |x| < 0.84375
        if ix < 0x3e30_0000 {
            // |x| < 2^-28
            if ix < 0x0080_0000 {
                // avoid underflow
                return 0.125 * (8.0 * x + EFX8 * x);
            }
            return x + EFX * x;
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        let y = r / s;
        return x + x * y;
    }
    if ix < 0x3ff4_0000 {
        // 0.84375 <= |x| < 1.25
        let s = x.abs() - 1.0;
        let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
        let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
        return if hx >= 0 { ERX + p / q } else { -ERX - p / q };
    }
    if ix >= 0x4018_0000 {
        // inf > |x| >= 6
        return if hx >= 0 { 1.0 - TINY } else { TINY - 1.0 };
    }
    let x = x.abs();
    let s = 1.0 / (x * x);
    let (r, big_s) = if ix < 0x4006_db6e {
        // |x| < 1/0.35
        (
            RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7)))))),
            1.0 + s * (SA1 + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8))))))),
        )
    } else {
        // |x| >= 1/0.35
        (
            RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6))))),
            1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7)))))),
        )
    };
    // z: x with its low word cleared (fdlibm's `__LO(z) = 0`), so z * z is exact
    let z = f64::from_bits(x.to_bits() & 0xffff_ffff_0000_0000);
    let r = exp(-z * z - 0.5625) * exp((z - x) * (z + x) + r / big_s);
    if hx >= 0 {
        1.0 - r / x
    } else {
        r / x - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::erf;

    #[test]
    fn special_values() {
        assert_eq!(erf(0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(erf(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erf(f64::NEG_INFINITY), -1.0);
        assert!(erf(f64::NAN).is_nan());
        assert_eq!(erf(6.0), 1.0);
        assert_eq!(erf(-30.0), -1.0);
    }

    #[test]
    fn odd() {
        for x in [1e-310, 1e-20, 0.3, 0.84375, 1.0, 1.25, 2.0, 1.0 / 0.35, 3.5, 5.9] {
            assert_eq!(erf(-x).to_bits(), (-erf(x)).to_bits(), "x = {x}");
        }
    }
}
