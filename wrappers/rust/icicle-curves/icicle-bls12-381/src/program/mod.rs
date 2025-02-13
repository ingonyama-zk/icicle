use crate::curve::ScalarField;

use icicle_core::impl_program_field;

impl_program_field!("bls12_381", bls12_381, ScalarField, ScalarCfg);
