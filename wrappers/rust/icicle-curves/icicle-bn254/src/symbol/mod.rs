use crate::curve::ScalarField;

use icicle_core::impl_invertible_symbol_ring;

impl_invertible_symbol_ring!("bn254", bn254, ScalarField);
