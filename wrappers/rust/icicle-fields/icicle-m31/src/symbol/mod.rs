use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_symbol_ring;

impl_symbol_ring!("m31", m31, ScalarField);
impl_symbol_ring!("m31_extension", m31_extension, ExtensionField);
