use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_symbol_ring;

impl_symbol_ring!("koalabear", koalabear, ScalarField);
impl_symbol_ring!("koalabear_extension", koalabear_extension, ExtensionField);
