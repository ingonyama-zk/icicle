use crate::curve::ScalarField;

use icicle_core::impl_program_ring;

impl_program_ring!("bn254", bn254, ScalarField);

#[cfg(test)]
use icicle_core::impl_program_tests;

#[cfg(test)]
impl_program_tests!(bn254, ScalarField);
