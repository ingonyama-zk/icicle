use icicle_bn254::curve::CurveCfg;

use icicle_core::impl_msm_bench;

impl_msm_bench!("bn254", CurveCfg);
