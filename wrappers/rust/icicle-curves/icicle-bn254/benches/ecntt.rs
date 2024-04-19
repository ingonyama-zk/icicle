#[cfg(feature = "ec_ntt")]
use icicle_bn254::curve::{BaseField, CurveCfg, ScalarField};

#[cfg(feature = "ec_ntt")]
use icicle_core::impl_ecntt_bench;
#[cfg(feature = "ec_ntt")]
impl_ecntt_bench!("bn254", ScalarField, BaseField, CurveCfg);

#[cfg(not(feature = "ec_ntt"))]
fn main() {}
