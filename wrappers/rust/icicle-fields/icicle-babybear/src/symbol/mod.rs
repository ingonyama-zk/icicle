use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_symbol_field;

impl_symbol_field!("babybear", babybear, ScalarField);
impl_symbol_field!("babybear_extension", babybear_extension, ExtensionField);
