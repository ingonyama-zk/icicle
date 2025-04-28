#[cfg(not(feature = "no_ecntt"))]
use icicle_bn254::curve::{Bn254ScalarField, CurveCfg};

#[cfg(not(feature = "no_ecntt"))]
use icicle_core::impl_ecntt_bench;
#[cfg(not(feature = "no_ecntt"))]
impl_ecntt_bench!("bn254", Bn254ScalarField, CurveCfg);

#[cfg(feature = "no_ecntt")]
fn main() {}
