use crate::curve::ScalarField;

use icicle_core::impl_program_ring;

impl_program_ring!("grumpkin", grumpkin, ScalarField);

#[cfg(test)]
use icicle_core::impl_program_tests;

#[cfg(test)]
impl_program_tests!(grumpkin, ScalarField);
