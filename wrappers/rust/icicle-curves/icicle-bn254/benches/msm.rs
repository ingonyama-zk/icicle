use icicle_bn254::curve::G1Projective;
use icicle_core::impl_msm_bench;

impl_msm_bench!("bn254", G1Projective);
