use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_symbol_field;

impl_symbol_field!("m31", m31, ScalarField, ScalarCfg);
impl_symbol_field!("m31_extension", m31_extension, ExtensionField, ExtensionCfg);
