use crate::curve::ScalarField;

use icicle_core::impl_program_ring;

impl_program_ring!("bw6_761", bw6_761, ScalarField);

#[cfg(test)]
use icicle_core::impl_program_tests;

#[cfg(test)]
impl_program_tests!(bw6_761, ScalarField);
