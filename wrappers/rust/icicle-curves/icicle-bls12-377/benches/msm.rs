use icicle_bls12_377::curve::CurveCfg;

use icicle_core::impl_msm_bench;

impl_msm_bench!("BLS12_377", CurveCfg);
