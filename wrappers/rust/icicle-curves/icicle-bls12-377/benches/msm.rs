use icicle_bls12_377::curve::Bls12377Curve;

use icicle_core::impl_msm_bench;

impl_msm_bench!("bls12_377", Bls12377Curve);
