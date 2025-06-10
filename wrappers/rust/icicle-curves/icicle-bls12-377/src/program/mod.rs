use crate::curve::ScalarField;

use icicle_core::impl_program_field;

impl_program_field!("bls12_377", bls12_377, ScalarField, ScalarCfg);
