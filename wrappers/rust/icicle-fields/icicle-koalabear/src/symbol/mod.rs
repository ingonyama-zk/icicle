use crate::field::{ScalarField, ExtensionField};

use icicle_core::impl_symbol_field;

impl_symbol_field!("koalabear", koalabear, ScalarField, ScalarCfg);
impl_symbol_field!("koalabear_extension", koalabear_extension, ExtensionField, ExtensionCfg);
