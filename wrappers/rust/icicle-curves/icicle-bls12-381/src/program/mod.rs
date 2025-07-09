use crate::curve::ScalarField;

use icicle_core::impl_program_ring;

impl_program_ring!("bls12_381", bls12_381, ScalarField);

#[cfg(test)]
use icicle_core::impl_program_tests;

#[cfg(test)]
impl_program_tests!(bls12_381, ScalarField);
