#[cfg(feature = "ec_ntt")]
use icicle_bn254::curve::{ScalarField, CurveCfg};

#[cfg(feature = "ec_ntt")]
use icicle_core::impl_ecntt_bench;
#[cfg(feature = "ec_ntt")]
impl_ecntt_bench!("bn254", ScalarField, CurveCfg);

#[cfg(not(feature = "ec_ntt"))]
fn main() {}
