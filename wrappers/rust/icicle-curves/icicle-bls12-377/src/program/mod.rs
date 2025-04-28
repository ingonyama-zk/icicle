use crate::curve::{Bls12_377BaseField, Bls12_377ScalarField};

use icicle_core::impl_program_field;

impl_program_field!("bls12_377", bls12_377, Bls12_377ScalarField);
#[cfg(feature = "bw6-761")]
impl_program_field!("bw6_761", bw6_761, Bls12_377BaseField);
