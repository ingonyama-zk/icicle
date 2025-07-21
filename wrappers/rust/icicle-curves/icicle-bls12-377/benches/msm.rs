use icicle_bls12_377::curve::G1Projective;

use icicle_core::impl_msm_bench;

impl_msm_bench!("bls12_377", G1Projective);
