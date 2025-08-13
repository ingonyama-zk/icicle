use crate::curve::ScalarField;

use icicle_core::impl_program_ring;

impl_program_ring!("bls12_377", bls12_377, ScalarField);

#[cfg(test)]
use icicle_core::impl_program_tests;

#[cfg(test)]
impl_program_tests!(bls12_377, ScalarField);
