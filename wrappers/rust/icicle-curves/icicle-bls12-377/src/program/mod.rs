use crate::curve::{ScalarField, BaseField};

use icicle_core::impl_program_field;

impl_program_field!("bls12_377", bls12_377, ScalarField, ScalarCfg);
#[cfg(feature = "bw6-761")]
impl_program_field!("bw6_761", bw6_761, BaseField, BaseCfg);
