use crate::field::{ScalarField, ExtensionField};

use icicle_core::impl_symbol_field;

impl_symbol_field!("babybear", babybear, ScalarField, ScalarCfg);
impl_symbol_field!("babybear_extension", babybear_extension, ExtensionField, ExtensionCfg);
