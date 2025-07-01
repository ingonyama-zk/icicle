use crate::curve::ScalarField;

use icicle_core::impl_program_field;

impl_program_field!("bn254", bn254, ScalarField, ScalarCfg);

#[cfg(test)]
use icicle_core::impl_program_tests;

#[cfg(test)]
impl_program_tests!(bn254, ScalarField);
