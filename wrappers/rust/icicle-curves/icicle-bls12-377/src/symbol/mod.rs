use crate::curve::Bls12_377ScalarField;

use icicle_core::impl_symbol_field;

impl_symbol_field!("bls12_377", bls12_377, Bls12_377ScalarField);
#[cfg(feature = "bw6-761")]
use crate::curve::Bls12_377BaseField;
#[cfg(feature = "bw6-761")]
impl_symbol_field!("bw6_761", bw6_761, Bls12_377BaseField);
