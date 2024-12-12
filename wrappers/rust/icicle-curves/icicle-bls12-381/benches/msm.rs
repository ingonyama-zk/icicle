use icicle_bls12_381::curve::Bls12381Curve;

use icicle_core::impl_msm_bench;

impl_msm_bench!("bls12_381", Bls12381Curve);
