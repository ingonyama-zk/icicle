use crate::curve::Bls12_381ScalarField;

use icicle_core::impl_program_field;

impl_program_field!("bls12_381", bls12_381, Bls12_381ScalarField);
