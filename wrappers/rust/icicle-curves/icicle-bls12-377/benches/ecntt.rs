use icicle_bls12_377::curve::{BaseField, CurveCfg, ScalarField};

use icicle_core::impl_ecntt_bench;

impl_ecntt_bench!("BLS12_377", ScalarField, BaseField, CurveCfg);
