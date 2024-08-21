#[cfg(not(feature = "no_ecntt"))]
use icicle_bn254::curve::{CurveCfg, ScalarField};

#[cfg(not(feature = "no_ecntt"))]
use icicle_core::impl_ecntt_bench;
#[cfg(not(feature = "no_ecntt"))]
impl_ecntt_bench!("bn254", ScalarField, CurveCfg);

#[cfg(not(feature = "ec_ntt"))]
fn main() {}
