#[cfg(not(feature = "no_ecntt"))]
use icicle_bls12_381::curve::{Bls12_381ScalarField, CurveCfg};

#[cfg(not(feature = "no_ecntt"))]
use icicle_core::impl_ecntt_bench;
#[cfg(not(feature = "no_ecntt"))]
impl_ecntt_bench!("bls12_381", Bls12_381ScalarField, CurveCfg);

#[cfg(feature = "no_ecntt")]
fn main() {}
