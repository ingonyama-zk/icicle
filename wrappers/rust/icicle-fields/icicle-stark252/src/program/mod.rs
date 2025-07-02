use crate::field::ScalarField;

use icicle_core::impl_program_field;

impl_program_field!("stark252", stark252, ScalarField);

#[cfg(test)]
use icicle_core::impl_program_tests;

#[cfg(test)]
impl_program_tests!(stark252, ScalarField);
