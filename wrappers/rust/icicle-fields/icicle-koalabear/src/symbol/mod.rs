use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_invertible_symbol_ring;

impl_invertible_symbol_ring!("koalabear", koalabear, ScalarField);
impl_invertible_symbol_ring!("koalabear_extension", koalabear_extension, ExtensionField);
