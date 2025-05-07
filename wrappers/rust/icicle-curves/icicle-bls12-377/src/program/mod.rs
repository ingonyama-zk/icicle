use crate::curve::ScalarField;

use icicle_core::impl_program_field;

impl_program_field!("bls12_377", bls12_377, ScalarField);
#[cfg(feature = "bw6-761")]
use crate::curve::BaseField;
#[cfg(feature = "bw6-761")]
impl_program_field!("bw6_761", bw6_761, BaseField);
