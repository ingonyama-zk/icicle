use icicle_bn254::curve::Bn254Curve;

use icicle_core::impl_msm_bench;

impl_msm_bench!("bn254", Bn254Curve);
