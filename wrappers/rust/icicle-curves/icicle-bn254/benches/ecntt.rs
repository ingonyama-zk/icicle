#[cfg(feature = "ecntt")]
use icicle_bn254::curve::{CurveCfg, ScalarField};

#[cfg(feature = "ecntt")]
use icicle_core::impl_ecntt_bench;
#[cfg(feature = "ecntt")]
impl_ecntt_bench!("bn254", ScalarField, CurveCfg);

#[cfg(not(feature = "ecntt"))]
fn main() {}
