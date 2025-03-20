use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_program_field;

impl_program_field!("babybear", babybear, ScalarField, ScalarCfg);
impl_program_field!("babybear_extension", babybear_extension, ExtensionField, ExtensionCfg);
