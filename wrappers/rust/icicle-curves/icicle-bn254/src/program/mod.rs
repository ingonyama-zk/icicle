use crate::curve::ScalarField;

use icicle_core::impl_program_field;

impl_program_field!("bn254", bn254, ScalarField, ScalarCfg);
