use crate::curve::ScalarField;

use icicle_core::impl_program_field;

impl_program_field!("bls12_377", bls12_377, ScalarField, ScalarCfg);

#[cfg(test)]
use icicle_core::impl_program_tests;

#[cfg(test)]
impl_program_tests!(bls12_377, ScalarField);
