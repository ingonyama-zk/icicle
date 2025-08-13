use crate::curve::ScalarField;

use icicle_core::impl_invertible_symbol_ring;

impl_invertible_symbol_ring!("bls12_381", bls12_381, ScalarField);
