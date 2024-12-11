#[cfg(not(feature = "no_ecntt"))]
use icicle_bls12_381::curve::{Bls12381Curve, ScalarField};

#[cfg(not(feature = "no_ecntt"))]
use icicle_core::impl_ecntt_bench;
#[cfg(not(feature = "no_ecntt"))]
impl_ecntt_bench!("bls12_381", ScalarField, Bls12381Curve);

#[cfg(feature = "no_ecntt")]
fn main() {}
