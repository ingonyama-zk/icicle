use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_program_field;

impl_program_field!("koalabear", koalabear, ScalarField, ScalarCfg);
impl_program_field!("koalabear_extension", koalabear_extension, ExtensionField, ExtensionCfg);
