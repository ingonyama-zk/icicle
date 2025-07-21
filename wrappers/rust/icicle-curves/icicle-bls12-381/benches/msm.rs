use icicle_bls12_381::curve::G1Projective;

use icicle_core::impl_msm_bench;

impl_msm_bench!("bls12_381", G1Projective);
