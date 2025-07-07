use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_invertible_symbol_ring;

impl_invertible_symbol_ring!("goldilocks", goldilocks, ScalarField);
impl_invertible_symbol_ring!("goldilocks_extension", goldilocks_extension, ExtensionField);
