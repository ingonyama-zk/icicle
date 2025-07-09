#[cfg(feature = "ecntt")]
use icicle_bls12_381::curve::{G1Projective, ScalarField};

#[cfg(feature = "ecntt")]
use icicle_core::impl_ecntt_bench;
#[cfg(feature = "ecntt")]
impl_ecntt_bench!("bls12_381", ScalarField, G1Projective);

#[cfg(not(feature = "ecntt"))]
fn main() {}
