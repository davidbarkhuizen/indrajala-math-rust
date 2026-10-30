//! Each kernel family's output bits, as a hash, against `fixtures/bits.txt`: every platform must
//! give the bits this file records (the crate's contract: a fixed grouping, the same bits on every
//! machine). Inputs come from the core's own PCG64, so they are the same everywhere too. After a
//! change that is meant to move bits (none so far), `INDRAJALA_MATH_BLESS=1 cargo test` rewrites the
//! file; the Python suite, which checks the same ops against numpy, says whether the new bits are
//! right.

use std::collections::BTreeMap;
use std::path::PathBuf;

use indrajala_math::conv::{conv_forward_batch, max_pool_forward_batch, ConvGeometry};
use indrajala_math::fused::layer_forward_batch;
use indrajala_math::generator::{Generator, SeedSequence};
use indrajala_math::layer_norm::layer_norm_forward_batch;
use indrajala_math::linalg::{matmul, matmul_add, matmul_long_k, matmul_nt, outer};
use indrajala_math::recurrent::gru_forward;
use indrajala_math::ufuncs::{array_softmax, exp};
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
    out.insert("softmax 7 x 10", hash(&[&array_softmax(&matrix(g, 7, 10))]));

    // conv and pooling on MNIST's shape
    let geometry = ConvGeometry::new(28, 28, 1, 3, 1).unwrap();
    let (w, x, b) = (matrix(g, 8, 9), matrix(g, 3, 784), vector(g, 8));
    let (a, cols) = conv_forward_batch(&w, &x, &b, &geometry).unwrap();
    out.insert("conv_forward_batch 28 x 28, 8 ch", hash(&[&a, &cols]));
    let pool = ConvGeometry::new(26, 26, 8, 2, 2).unwrap();
    let (pooled, argmax) = max_pool_forward_batch(&a, &pool).unwrap();
    out.insert("max_pool_forward_batch 26 x 26, 8 ch", hash(&[&pooled, &argmax]));

    // dense, layer norm and a GRU cell
    let (w, x, b) = (matrix(g, 30, 784), matrix(g, 5, 784), vector(g, 30));
    out.insert(
        "layer_forward_batch 30 x 784, b5",
        hash(&[&layer_forward_batch(&w, &x, &b).unwrap()]),
    );
    let (x, gamma, beta) = (matrix(g, 4, 3 * 8), vector(g, 8), vector(g, 8));
    let (a, xhat, std) = layer_norm_forward_batch(&x, &gamma, &beta, 1e-5).unwrap();
    out.insert("layer_norm_forward_batch 4 x 3 tokens x 8", hash(&[&a, &xhat, &std]));
    let (u, c) = (matrix(g, 4, 6), matrix(g, 4, 5));
    let (w, uw, b) = (matrix(g, 15, 6), matrix(g, 15, 5), vector(g, 15));
    let (h, r, z, h_tilde, rc) = gru_forward(&u, &c, &w, &uw, &b).unwrap();
    out.insert("gru_forward 4 x 6, state 5", hash(&[&h, &r, &z, &h_tilde, &rc]));

    // the generator itself
    out.insert("Generator uniform 1000", hash(&[&vector(&mut rng(7), 1000)]));
    out
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
