#[cfg(not(feature = "no_ecntt"))]
use icicle_bls12_377::curve::{Bls12377Curve, ScalarField};

#[cfg(not(feature = "no_ecntt"))]
use icicle_core::impl_ecntt_bench;
#[cfg(not(feature = "no_ecntt"))]
impl_ecntt_bench!("bls12_377", ScalarField, Bls12377Curve);

#[cfg(feature = "no_ecntt")]
fn main() {}
