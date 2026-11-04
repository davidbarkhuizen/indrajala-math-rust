//! Each kernel family's output bits, as a hash, against `fixtures/bits.txt`: every platform must
//! give the bits this file records (the crate's contract: a fixed grouping, the same bits on every
//! machine). Inputs come from the core's own PCG64, so they are the same everywhere too. After a
//! change that is meant to move bits (none so far), `INDRAJALA_MATH_BLESS=1 cargo test` rewrites the
//! file; the Python suite, which checks the same ops against numpy, says whether the new bits are
//! right.

use std::collections::BTreeMap;
use std::path::PathBuf;

use indrajala_math::linalg::{matmul, matmul_add, matmul_long_k, matmul_nt, matmul_transposed_add, outer};
use indrajala_math::nn::attention::{self, cross};
use indrajala_math::nn::conv::{self, max_pool_forward_batch};
use indrajala_math::nn::norm::layer_norm;
use indrajala_math::nn::recurrent::gru_forward;
use indrajala_math::nn::{dense, gated};
use indrajala_math::num::activation::Activation;
use indrajala_math::ops::unary::{self, erf, exp};
use indrajala_math::random::generator::{Generator, SeedSequence};
use indrajala_math::{Array, Shape};

/// FNV-1a over the values' bits, and their count.
fn hash(arrays: &[&Array]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut count = 0;
    for a in arrays {
        for v in a.data() {
            for byte in v.to_bits().to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            count += 1;
        }
    }
    format!("{h:016x} {count}")
}

fn rng(seed: u32) -> Generator {
    Generator::from_seed_sequence(&SeedSequence::from_words(&[seed], &[]))
}

fn matrix(rng: &mut Generator, rows: usize, cols: usize) -> Array {
    rng.uniform(-1.0, 1.0, Shape::Matrix(rows, cols)).unwrap()
}

fn vector(rng: &mut Generator, n: usize) -> Array {
    rng.uniform(-1.0, 1.0, Shape::Vector(n)).unwrap()
}

fn families() -> BTreeMap<&'static str, String> {
    let mut out = BTreeMap::new();
    let g = &mut rng(20261030);

    // linalg: each shape case, the register tiles and their tails, and a threaded product
    let (w, x) = (matrix(g, 30, 784), vector(g, 784));
    out.insert("matmul matrix @ vector 30 x 784", hash(&[&matmul(&w, &x).unwrap()]));
    let (d, w2) = (vector(g, 30), matrix(g, 30, 784));
    out.insert("matmul vector @ matrix 30 x 784", hash(&[&matmul(&d, &w2).unwrap()]));
    let (a, b) = (matrix(g, 37, 29), matrix(g, 29, 13));
    out.insert("matmul 37 x 29 @ 29 x 13", hash(&[&matmul(&a, &b).unwrap()]));
    let (a, b) = (matrix(g, 256, 256), matrix(g, 256, 256));
    out.insert(
        "matmul 256 x 256 @ 256 x 256 (threaded)",
        hash(&[&matmul(&a, &b).unwrap()]),
    );
    let (c, a, b) = (matrix(g, 30, 13), matrix(g, 30, 33), matrix(g, 33, 13));
    out.insert(
        "matmul_add 30 x 33 @ 33 x 13",
        hash(&[&matmul_add(&c, &a, &b).unwrap()]),
    );
    let (a, b) = (matrix(g, 33, 5408), matrix(g, 31, 5408));
    out.insert("matmul_nt 33 x 5408, 31 x 5408", hash(&[&matmul_nt(&a, &b).unwrap()]));
    let (a, b) = (matrix(g, 8, 4100), matrix(g, 4100, 75));
    out.insert(
        "matmul_long_k 8 x 4100 @ 4100 x 75",
        hash(&[&matmul_long_k(&a, &b).unwrap()]),
    );
    let t = matrix(g, 515, 31);
    out.insert("transpose 515 x 31", hash(&[&t.transpose()]));
    let (u, v) = (vector(g, 17), vector(g, 23));
    out.insert("outer 17 x 23", hash(&[&outer(&u, &v).unwrap()]));

    // exp's fast path, its vector lanes and their tails, and the inputs `exp` redoes
    let mut values: Vec<f64> = g.uniform(-745.0, 710.0, Shape::Vector(1001)).unwrap().into_data();
    values.extend([
        0.0,
        -0.0,
        1e-300,
        -1e-300,
        709.78,
        -745.1,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ]);
    out.insert("exp", hash(&[&exp(&Array::from_vector(values))]));
    out.insert("softmax 7 x 10", hash(&[&unary::softmax(&matrix(g, 7, 10))]));

    // conv and pooling on MNIST's shape
    let geometry = conv::Geometry::new(28, 28, 1, 3, 1).unwrap();
    let (w, x, b) = (matrix(g, 8, 9), matrix(g, 3, 784), vector(g, 8));
    let (a, cols) = conv::forward_batch(&w, &x, &b, &geometry).unwrap();
    out.insert("conv_forward_batch 28 x 28, 8 ch", hash(&[&a, &cols]));
    let pool = conv::Geometry::new(26, 26, 8, 2, 2).unwrap();
    let (pooled, argmax) = max_pool_forward_batch(&a, &pool).unwrap();
    out.insert("max_pool_forward_batch 26 x 26, 8 ch", hash(&[&pooled, &argmax]));

    // dense, layer norm and a GRU cell
    let (w, x, b) = (matrix(g, 30, 784), matrix(g, 5, 784), vector(g, 30));
    out.insert(
        "layer_forward_batch 30 x 784, b5",
        hash(&[&dense::forward_batch(&w, &x, &b).unwrap()]),
    );
    let (x, gamma, beta) = (matrix(g, 4, 3 * 8), vector(g, 8), vector(g, 8));
    let (a, xhat, std) = layer_norm::forward_batch(&x, &gamma, &beta, 1e-5).unwrap();
    out.insert("layer_norm_forward_batch 4 x 3 tokens x 8", hash(&[&a, &xhat, &std]));
    let (u, c) = (matrix(g, 4, 6), matrix(g, 4, 5));
    let (w, uw, b) = (matrix(g, 15, 6), matrix(g, 15, 5), vector(g, 15));
    let (h, r, z, h_tilde, rc) = gru_forward(&u, &c, &w, &uw, &b).unwrap();
    out.insert("gru_forward 4 x 6, state 5", hash(&[&h, &r, &z, &h_tilde, &rc]));

    // the generator itself
    out.insert("Generator uniform 1000", hash(&[&vector(&mut rng(7), 1000)]));

    // erf's four ranges and their boundaries, both signs, and the special values
    let mut values: Vec<f64> = g.uniform(-7.0, 7.0, Shape::Vector(1001)).unwrap().into_data();
    values.extend([
        0.0,
        -0.0,
        1e-310,
        -1e-20,
        0.84375,
        -1.25,
        1.0 / 0.35,
        6.0,
        -6.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ]);
    out.insert("erf", hash(&[&erf(&Array::from_vector(values))]));

    // each activation's fused forward (a and the kept z) and hidden delta, through the register
    // tiles' tails, over pre-activations wide enough to reach erf's ranges and tanh's saturation
    let (w, x, b) = (matrix(g, 37, 29), matrix(g, 13, 29), vector(g, 37));
    let x = Array::from_matrix(x.data().iter().map(|v| v * 4.0).collect(), 13, 29);
    let (next_w, next_delta) = (matrix(g, 11, 37), matrix(g, 13, 11));
    for activation in [
        Activation::Relu,
        Activation::Gelu,
        Activation::GeluTanh,
        Activation::Silu,
    ] {
        let (a, z) = dense::activation_forward_batch(&w, &x, &b, activation).unwrap();
        let cached = z.as_ref().unwrap_or(&a);
        let delta = dense::activation_hidden_delta_batch(&next_w, &next_delta, cached, activation).unwrap();
        let mut arrays = vec![&a, &delta];
        arrays.extend(z.as_ref());
        out.insert(
            match activation {
                Activation::Relu => "activation relu forward and hidden delta 13 x 29 -> 37",
                Activation::Gelu => "activation gelu forward and hidden delta 13 x 29 -> 37",
                Activation::GeluTanh => "activation gelu_tanh forward and hidden delta 13 x 29 -> 37",
                Activation::Silu => "activation silu forward and hidden delta 13 x 29 -> 37",
                Activation::Sigmoid => unreachable!("batch norm's, not a hidden layer's"),
            },
            hash(&arrays),
        );
    }

    // the gated FFN's forward (out and z) and delta, SwiGLU and GEGLU, through the tiles' tails
    let (w, x, b) = (matrix(g, 34, 29), matrix(g, 13, 29), vector(g, 34));
    let x = Array::from_matrix(x.data().iter().map(|v| v * 4.0).collect(), 13, 29);
    let delta = matrix(g, 13, 17);
    for (name, activation) in [("silu", Activation::Silu), ("gelu", Activation::Gelu)] {
        let (y, z) = gated::forward_batch(&w, &x, &b, activation).unwrap();
        let dz = gated::delta_batch(&delta, &z, activation).unwrap();
        let label = match name {
            "silu" => "gated silu forward and delta 13 x 29 -> 17",
            _ => "gated gelu forward and delta 13 x 29 -> 17",
        };
        out.insert(label, hash(&[&y, &z, &dz]));
    }

    // attention's forward and downstream, causal and rotary, over three key blocks, at 4 query heads
    // of 4 with their own key/value heads, two groups and one (indrajala-ml's grouped-query
    // attention workplan, D4)
    let (examples, tokens, d, heads, d_k) = (2, 37, 8, 4, 4);
    let x = matrix(g, examples, tokens * d);
    let delta = matrix(g, examples, tokens * d);
    let (cos, sin) = (matrix(g, tokens, d_k / 2), matrix(g, tokens, d_k / 2));
    for key_value_heads in [4, 2, 1] {
        let (width, kv_width) = (heads * d_k, key_value_heads * d_k);
        let (wq, bq) = (matrix(g, width, d), vector(g, width));
        let (wk, bk, wv, bv) = (
            matrix(g, kv_width, d),
            vector(g, kv_width),
            matrix(g, kv_width, d),
            vector(g, kv_width),
        );
        let (wo, bo) = (matrix(g, d, width), vector(g, d));
        let (a, q, k, v, h, m, l, _) = attention::forward_batch(
            &x,
            &wq,
            &bq,
            &wk,
            &bk,
            &wv,
            &bv,
            &wo,
            &bo,
            heads,
            key_value_heads,
            true,
            0.0,
            false,
            None,
            Some(&cos),
            Some(&sin),
            None,
            16,
        )
        .unwrap();
        let (dx, dq, dk, dv) = attention::downstream_batch(
            &delta,
            &wq,
            &wk,
            &wv,
            &wo,
            [&q, &k, &v, &h, &m, &l],
            heads,
            key_value_heads,
            true,
            None,
            0.0,
            None,
            Some(&cos),
            Some(&sin),
            16,
        )
        .unwrap();
        out.insert(
            match key_value_heads {
                4 => "attention forward and downstream 2 x 37 x 8, 4 heads",
                2 => "attention forward and downstream 2 x 37 x 8, 4 heads sharing 2",
                _ => "attention forward and downstream 2 x 37 x 8, 4 heads sharing 1",
            },
            hash(&[&a, &q, &k, &v, &h, &m, &l, &dx, &dq, &dk, &dv]),
        );
    }

    // cross-attention's forward and downstream (indrajala-ml's cross-attention workplan, D3, D13):
    // 29 queries of 8 features on a memory of 37 tokens of 12, over three key blocks, 4 query heads
    // sharing 2, with packed pairs' segments on both sides and pads, so some queries read no key
    let (memory_tokens, d_m, key_value_heads, queries) = (37, 12, 2, 29);
    let x = matrix(g, examples, queries * d);
    let memory = matrix(g, examples, memory_tokens * d_m);
    let delta = matrix(g, examples, queries * d);
    let (width, kv_width) = (heads * d_k, key_value_heads * d_k);
    let (wq, bq) = (matrix(g, width, d), vector(g, width));
    let (wk, bk) = (matrix(g, kv_width, d_m), vector(g, kv_width));
    let (wv, bv) = (matrix(g, kv_width, d_m), vector(g, kv_width));
    let (wo, bo) = (matrix(g, d, width), vector(g, d));
    let ids = |pads: usize, lengths: &[usize], tokens: usize| {
        let mut row = vec![0.0; pads];
        for (k, &n) in lengths.iter().enumerate() {
            row.extend(std::iter::repeat_n((k + 1) as f64, n));
        }
        row.resize(tokens, 0.0);
        row
    };
    let mut segments = ids(3, &[10, 6, 10], queries);
    segments.extend(ids(0, &[20, 9], queries));
    let mut memory_segments = ids(0, &[12, 15, 10], memory_tokens);
    memory_segments.extend(ids(5, &[14, 18], memory_tokens));
    let (segments, memory_segments) = (
        Array::from_matrix(segments, examples, queries),
        Array::from_matrix(memory_segments, examples, memory_tokens),
    );
    let (a, q, k, v, h, m, l, _) = cross::forward_batch(
        &x,
        &memory,
        &wq,
        &bq,
        &wk,
        &bk,
        &wv,
        &bv,
        &wo,
        &bo,
        heads,
        key_value_heads,
        0.0,
        false,
        None,
        Some(&segments),
        Some(&memory_segments),
        16,
    )
    .unwrap();
    let (dx, dm, dq, dk, dv) = cross::downstream_batch(
        &delta,
        &wq,
        &wk,
        &wv,
        &wo,
        [&q, &k, &v, &h, &m, &l],
        heads,
        key_value_heads,
        Some(&segments),
        Some(&memory_segments),
        0.0,
        None,
        16,
    )
    .unwrap();
    out.insert(
        "cross-attention forward and downstream 2 x 29 x 8 on 37 x 12, 4 heads sharing 2",
        hash(&[&a, &q, &k, &v, &h, &m, &l, &dx, &dm, &dq, &dk, &dv]),
    );

    // long k, each past a block of `matmul_2d`'s slabs and on either side of one: its own generator,
    // so these inputs don't move the families above
    let g = &mut rng(20261104);
    for (m, k, n) in [
        (5, 64, 21),
        (5, 65, 21),
        (3, 300, 37),
        (9, 1025, 18),
        (64, 2048, 16),
        (7, 512, 800),
        (3, 100, 517),
        (33, 80, 530),
        (64, 100, 37),
        (70, 300, 21),
        (130, 70, 512),
    ] {
        let (a, b, c) = (matrix(g, m, k), matrix(g, k, n), matrix(g, m, n));
        // a leak for the family's name: a test, and these few, once
        let name: &'static str = Box::leak(format!("matmul long k {m} x {k} x {n}").into_boxed_str());
        let name_add: &'static str = Box::leak(format!("matmul_add long k {m} x {k} x {n}").into_boxed_str());
        out.insert(name, hash(&[&matmul(&a, &b).unwrap()]));
        out.insert(name_add, hash(&[&matmul_add(&c, &a, &b).unwrap()]));
    }

    // a @ b.T past each machine's L2 share, with row and column tails and a `k` tail
    let g = &mut rng(20261105);
    for (m, k, n) in [(13, 2051, 35), (64, 5408, 32), (3, 8197, 7), (9, 1299, 501)] {
        let (a, b) = (matrix(g, m, k), matrix(g, n, k));
        let name: &'static str = Box::leak(format!("matmul_nt past L2 {m} x {k} x {n}").into_boxed_str());
        out.insert(name, hash(&[&matmul_nt(&a, &b).unwrap()]));
    }
    out
}

/// Reading `d` through steps changes no chain, on each side of the slab, packing and threading thresholds.
#[test]
fn matmul_transposed_add_is_matmul_add_of_the_transpose() {
    let g = &mut rng(20261106);
    for (m, k, n) in [
        (1, 1, 1),
        (3, 7, 5),
        (30, 33, 13),
        (16, 64, 35),
        (32, 32, 5408),
        (30, 65, 17),
        (64, 100, 37),
        (3, 100, 517),
        (70, 300, 21),
        (130, 70, 512),
        (256, 256, 256),
    ] {
        let (c, d, b) = (matrix(g, m, n), matrix(g, k, m), matrix(g, k, n));
        let direct = matmul_transposed_add(&c, &d, &b).unwrap();
        let through_the_copy = matmul_add(&c, &d.transpose(), &b).unwrap();
        assert_eq!(hash(&[&direct]), hash(&[&through_the_copy]), "{m} x {k} x {n}");
    }
}

#[test]
fn every_family_gives_the_recorded_bits() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bits.txt");
    let got = families();
    let text: String = got.iter().map(|(name, h)| format!("{name}\t{h}\n")).collect();
    if std::env::var_os("INDRAJALA_MATH_BLESS").is_some() {
        std::fs::write(&path, &text).unwrap();
        return;
    }
    let recorded = std::fs::read_to_string(&path).expect("tests/fixtures/bits.txt (INDRAJALA_MATH_BLESS=1 writes it)");
    let mut moved = Vec::new();
    for line in recorded.lines() {
        let (name, h) = line.split_once('\t').unwrap();
        match got.get(name) {
            Some(g) if g == h => {}
            Some(g) => moved.push(format!("{name}: recorded {h}, got {g}")),
            None => moved.push(format!("{name}: recorded, not computed")),
        }
    }
    assert_eq!(
        recorded.lines().count(),
        got.len(),
        "the fixture and the families differ in count"
    );
    assert!(moved.is_empty(), "bits moved:\n{}", moved.join("\n"));
}
