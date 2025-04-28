use crate::curve::Bn254ScalarField;

use icicle_core::impl_program_field;

impl_program_field!("bn254", bn254, Bn254ScalarField);
