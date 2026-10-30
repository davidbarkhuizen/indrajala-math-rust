//! The two generators against numpy's own published draws: `np.random.seed(42);
//! np.random.random(3)` and `np.random.default_rng(42).random(3)`. Its own test binary: the global
//! generator is process-wide.

use indrajala_math::generator::{Generator, SeedSequence};
use indrajala_math::{random, Shape};

#[test]
fn the_global_generator_is_numpys_legacy_stream() {
    random::seed_word(42);
    let draws = random::random(Shape::Vector(3));
    assert_eq!(draws.data(), &[0.3745401188473625, 0.9507143064099162, 0.7319939418114051]);
}

#[test]
fn default_rng_is_numpys_pcg64_stream() {
    let mut generator = Generator::from_seed_sequence(&SeedSequence::from_words(&[42], &[]));
    let draws = generator.random(Shape::Vector(3));
    assert_eq!(draws.data(), &[0.7739560485559633, 0.4388784397520523, 0.8585979199113825]);
}
