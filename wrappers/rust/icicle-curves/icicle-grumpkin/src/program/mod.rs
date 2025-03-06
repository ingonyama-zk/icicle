use crate::curve::ScalarField;

use icicle_core::impl_program_field;

impl_program_field!("grumpkin", grumpkin, ScalarField, ScalarCfg);
