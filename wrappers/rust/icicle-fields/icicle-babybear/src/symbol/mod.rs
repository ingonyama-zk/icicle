use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_invertible_symbol_ring;

impl_invertible_symbol_ring!("babybear", babybear, ScalarField);
impl_invertible_symbol_ring!("babybear_extension", babybear_extension, ExtensionField);
