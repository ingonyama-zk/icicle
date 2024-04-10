use icicle_bn254::curve::{BaseField, CurveCfg, ScalarField};

use icicle_core::impl_ecntt_bench;

impl_ecntt_bench!("BN254", ScalarField, BaseField, CurveCfg);
