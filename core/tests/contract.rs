//! The API's contract: bad arguments come back as `Err`, with the message the Python bindings
//! raise; only the constructors documented to panic do.

use indrajala_math::conv::ConvGeometry;
use indrajala_math::generator::{Generator, SeedSequence};
use indrajala_math::linalg::{matmul, matmul_nt};
use indrajala_math::{random, Array, Error, Shape};

#[test]
fn an_array_holds_its_shapes_count() {
    assert!(Array::new(vec![0.0; 6], Shape::Matrix(2, 3)).is_ok());
    let error = Array::new(vec![0.0; 5], Shape::Matrix(2, 3)).unwrap_err();
    assert_eq!(
        error.to_string(),
        "an array of shape Matrix(2, 3) holds 6 values, got 5"
    );
}

#[test]
#[should_panic(expected = "from_matrix")]
fn from_matrix_panics_on_a_short_buffer() {
    Array::from_matrix(vec![0.0; 5], 2, 3);
}

#[test]
fn a_product_of_mismatched_shapes_is_an_error() {
    let (a, b) = (
        Array::from_matrix(vec![1.0; 6], 2, 3),
        Array::from_matrix(vec![1.0; 6], 2, 3),
    );
    assert_eq!(
        matmul(&a, &b).unwrap_err(),
        Error::InvalidArgument("cannot matmul arrays of shape Matrix(2, 3) and Matrix(2, 3)".to_string())
    );
    assert!(matmul_nt(&a, &Array::from_vector(vec![1.0; 3])).is_err());
}

#[test]
fn a_geometry_is_checked() {
    assert!(ConvGeometry::new(28, 28, 1, 3, 1).is_ok());
    assert!(matches!(
        ConvGeometry::new(2, 2, 1, 3, 1),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        ConvGeometry::new(28, 28, 1, 3, 0),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn a_range_out_of_bounds_is_an_overflow() {
    assert!(matches!(
        random::uniform(f64::MIN, f64::MAX, Shape::Vector(1)),
        Err(Error::Overflow(_))
    ));
    let mut generator = Generator::from_seed_sequence(&SeedSequence::from_words(&[1], &[]));
    assert!(matches!(
        generator.uniform(1.0, 0.0, Shape::Vector(1)),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn exp_into_and_in_place_give_exps_bits() {
    let values: Vec<f64> = (0..11).map(|i| f64::from(i) * 0.7 - 3.0).collect();
    let mut out = vec![0.0; values.len()];
    indrajala_math::cr_exp::exp_into(&values, &mut out);
    let mut in_place = values.clone();
    indrajala_math::cr_exp::exp_in_place(&mut in_place);
    for ((&x, &a), &b) in values.iter().zip(&out).zip(&in_place) {
        let e = indrajala_math::cr_exp::exp(x);
        assert_eq!((a.to_bits(), b.to_bits()), (e.to_bits(), e.to_bits()));
    }
}

#[test]
fn small_products_and_a_transpose_run_their_scalar_paths() {
    let a = Array::from_matrix((0..12).map(f64::from).collect(), 3, 4);
    let b = Array::from_matrix((0..20).map(f64::from).collect(), 4, 5);
    assert_eq!(matmul(&a, &b).unwrap().shape(), Shape::Matrix(3, 5));
    assert_eq!(matmul_nt(&a, &a).unwrap().shape(), Shape::Matrix(3, 3));
    assert_eq!(a.transpose().transpose(), a);
}
