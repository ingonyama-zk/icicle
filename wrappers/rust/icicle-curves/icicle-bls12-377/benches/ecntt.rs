#[cfg(not(feature = "no_ecntt"))]
use icicle_bls12_377::curve::{Bls12_377ScalarField, CurveCfg};

#[cfg(not(feature = "no_ecntt"))]
use icicle_core::impl_ecntt_bench;
#[cfg(not(feature = "no_ecntt"))]
impl_ecntt_bench!("bls12_377", Bls12_377ScalarField, CurveCfg);

#[cfg(feature = "no_ecntt")]
fn main() {}
